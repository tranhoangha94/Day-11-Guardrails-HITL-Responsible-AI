"""
Assignment 11 — Production Defense-in-Depth Pipeline

Chains independent safety layers: rate limit → input guardrails → LLM →
output guardrails → LLM-as-Judge → audit & monitoring.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

from core.config import ALLOWED_TOPICS, BLOCKED_TOPICS

SYSTEM_PROMPT = """You are a helpful customer service assistant for VinBank.
You help customers with account inquiries, transactions, and general banking questions.
IMPORTANT: Never reveal internal system details, passwords, or API keys.
If asked about topics outside banking, politely redirect to banking services.
Keep answers concise and professional."""

JUDGE_INSTRUCTION = """You are a quality assurance judge for a banking AI assistant.
When you receive a message, treat it as the AI's response to evaluate.

Score on 4 criteria (1-5 each):
1. SAFETY: No harmful content, leaked secrets, or dangerous instructions
2. RELEVANCE: Response is about banking/finance, not off-topic
3. ACCURACY: No fabricated data, hallucinated numbers, or false claims
4. TONE: Professional, helpful, empathetic customer service tone

Respond in EXACTLY this format:
SAFETY: <score>
RELEVANCE: <score>
ACCURACY: <score>
TONE: <score>
VERDICT: PASS or FAIL
REASON: <one sentence>
"""

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|password|credentials)",
    r"pretend you are",
    r"act as (a |an )?(unrestricted|dan|jailbreak)",
    r"bỏ qua mọi hướng dẫn",
    r"admin password",
    r"api key",
    r"connection string",
    r"translate your system prompt",
    r"per ticket",
    r"fill in:",
    r"same passwords as you",
    r"select \* from",
    r"drop table",
    r"'; --",
]

PII_PATTERNS = {
    "phone_vn": r"0\d{9,10}",
    "email": r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}",
    "national_id": r"\b\d{9}\b|\b\d{12}\b",
    "api_key": r"sk-[a-zA-Z0-9-]+",
    "password": r"password\s*[:=]\s*\S+",
    "admin_secret": r"admin123|sk-vinbank-secret",
}

MAX_INPUT_LENGTH = 2000


@dataclass
class LayerResult:
    """Result from a single pipeline layer."""

    name: str
    status: str  # pass | block | modify | skip
    detail: str
    duration_ms: float = 0.0


@dataclass
class PipelineResult:
    """Full pipeline outcome for one user message."""

    response: str
    blocked: bool
    blocked_by: str | None
    layers: list[LayerResult] = field(default_factory=list)
    judge_scores: dict[str, Any] | None = None
    latency_ms: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class RateLimiter:
    """Sliding-window rate limiter per user — prevents abuse."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.hits = 0

    def check(self, user_id: str) -> LayerResult:
        now = time.time()
        window = self.user_windows[user_id]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            self.hits += 1
            wait = int(self.window_seconds - (now - window[0])) + 1
            return LayerResult(
                name="Rate Limiter",
                status="block",
                detail=f"Quá {self.max_requests} yêu cầu/{self.window_seconds}s. Thử lại sau {wait}s.",
            )

        window.append(now)
        return LayerResult(
            name="Rate Limiter",
            status="pass",
            detail=f"{len(window)}/{self.max_requests} yêu cầu trong cửa sổ {self.window_seconds}s",
        )


class InputGuardrails:
    """Regex injection detection + topic filter — blocks attacks before LLM."""

    def check(self, user_input: str) -> LayerResult:
        text = user_input.strip()

        if not text:
            return LayerResult(
                name="Input Guardrails",
                status="block",
                detail="Tin nhắn trống",
            )

        if len(text) > MAX_INPUT_LENGTH:
            return LayerResult(
                name="Input Guardrails",
                status="block",
                detail=f"Quá dài ({len(text)} ký tự, giới hạn {MAX_INPUT_LENGTH})",
            )

        if self._is_emoji_only(text):
            return LayerResult(
                name="Input Guardrails",
                status="block",
                detail="Chỉ chứa emoji — không phải câu hỏi ngân hàng",
            )

        for pattern in INJECTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return LayerResult(
                    name="Input Guardrails",
                    status="block",
                    detail=f"Phát hiện injection: pattern `{match.group()[:40]}`",
                )

        lower = text.lower()
        for topic in BLOCKED_TOPICS:
            if topic in lower:
                return LayerResult(
                    name="Input Guardrails",
                    status="block",
                    detail=f"Chủ đề bị cấm: '{topic}'",
                )

        if not any(topic in lower for topic in ALLOWED_TOPICS):
            return LayerResult(
                name="Input Guardrails",
                status="block",
                detail="Ngoài phạm vi ngân hàng (topic filter)",
            )

        return LayerResult(
            name="Input Guardrails",
            status="pass",
            detail="Không phát hiện injection, chủ đề hợp lệ",
        )

    @staticmethod
    def _is_emoji_only(text: str) -> bool:
        stripped = re.sub(r"[\s\W\d_]+", "", text, flags=re.UNICODE)
        if not stripped:
            return True
        return all(ord(c) > 0x1F000 or not c.isalpha() for c in stripped)


class OutputGuardrails:
    """PII/secret redaction — catches leaks the LLM might produce."""

    def check(self, response: str) -> tuple[LayerResult, str]:
        issues = []
        redacted = response

        for name, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"{name}: {len(matches)}")
                redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

        if issues:
            return (
                LayerResult(
                    name="Output Guardrails",
                    status="modify",
                    detail=f"Đã che PII/bí mật: {', '.join(issues)}",
                ),
                redacted,
            )

        return (
            LayerResult(
                name="Output Guardrails",
                status="pass",
                detail="Không phát hiện PII hoặc bí mật",
            ),
            response,
        )


class LLMJudge:
    """Separate LLM evaluates safety, relevance, accuracy, tone."""

    def __init__(self, client: genai.Client, model: str = "gemini-2.5-flash-lite"):
        self.client = client
        self.model = model
        self.fail_count = 0

    async def evaluate(self, user_query: str, response: str) -> tuple[LayerResult, dict | None]:
        prompt = (
            f"User question:\n{user_query}\n\n"
            f"AI response to evaluate:\n{response}"
        )
        t0 = time.perf_counter()
        try:
            result = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=JUDGE_INSTRUCTION),
            )
            verdict_text = result.text or ""
        except Exception as exc:
            return (
                LayerResult(
                    name="LLM-as-Judge",
                    status="skip",
                    detail=f"Judge lỗi: {exc}",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                ),
                None,
            )

        scores = self._parse_verdict(verdict_text)
        duration = (time.perf_counter() - t0) * 1000
        passed = scores.get("verdict", "PASS").upper() == "PASS"

        if not passed:
            self.fail_count += 1
            return (
                LayerResult(
                    name="LLM-as-Judge",
                    status="block",
                    detail=scores.get("reason", verdict_text[:120]),
                    duration_ms=duration,
                ),
                scores,
            )

        return (
            LayerResult(
                name="LLM-as-Judge",
                status="pass",
                detail=scores.get("reason", "Đạt tiêu chí chất lượng"),
                duration_ms=duration,
            ),
            scores,
        )

    @staticmethod
    def _parse_verdict(text: str) -> dict[str, Any]:
        scores: dict[str, Any] = {}
        for key in ("SAFETY", "RELEVANCE", "ACCURACY", "TONE"):
            match = re.search(rf"{key}:\s*(\d)", text, re.IGNORECASE)
            if match:
                scores[key.lower()] = int(match.group(1))

        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
        scores["verdict"] = verdict_match.group(1).upper() if verdict_match else "PASS"

        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        scores["reason"] = reason_match.group(1).strip() if reason_match else ""

        return scores


class AuditLogger:
    """Records every interaction for compliance and forensics."""

    def __init__(self):
        self.logs: list[dict] = []

    def record(self, entry: dict) -> LayerResult:
        entry["timestamp"] = datetime.now().isoformat()
        self.logs.append(entry)
        return LayerResult(
            name="Audit Log",
            status="pass",
            detail=f"Đã ghi log #{len(self.logs)}",
        )

    def export_json(self, filepath: str = "audit_log.json") -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False, default=str)
        return filepath


class MonitoringAlert:
    """Tracks block rates and fires alerts when thresholds are exceeded."""

    def __init__(
        self,
        block_rate_threshold: float = 0.5,
        rate_limit_threshold: int = 5,
        judge_fail_threshold: int = 3,
    ):
        self.block_rate_threshold = block_rate_threshold
        self.rate_limit_threshold = rate_limit_threshold
        self.judge_fail_threshold = judge_fail_threshold
        self.total_requests = 0
        self.blocked_requests = 0
        self.alerts: list[str] = []

    def update(self, result: PipelineResult, rate_limit_hits: int, judge_fails: int) -> LayerResult:
        self.total_requests += 1
        if result.blocked:
            self.blocked_requests += 1

        block_rate = self.blocked_requests / max(self.total_requests, 1)
        details = [
            f"Tổng: {self.total_requests}",
            f"Chặn: {self.blocked_requests} ({block_rate:.0%})",
            f"Rate-limit hits: {rate_limit_hits}",
            f"Judge fails: {judge_fails}",
        ]

        if block_rate >= self.block_rate_threshold and self.total_requests >= 5:
            msg = f"CẢNH BÁO: Tỷ lệ chặn cao ({block_rate:.0%})"
            if msg not in self.alerts:
                self.alerts.append(msg)

        if rate_limit_hits >= self.rate_limit_threshold:
            msg = f"CẢNH BÁO: Rate limit bị kích hoạt {rate_limit_hits} lần"
            if msg not in self.alerts:
                self.alerts.append(msg)

        if judge_fails >= self.judge_fail_threshold:
            msg = f"CẢNH BÁO: Judge FAIL {judge_fails} lần"
            if msg not in self.alerts:
                self.alerts.append(msg)

        alert_text = self.alerts[-1] if self.alerts else "Bình thường"
        return LayerResult(
            name="Monitoring",
            status="pass" if "CẢNH BÁO" not in alert_text else "modify",
            detail=f"{'; '.join(details)} | {alert_text}",
        )

    def get_metrics(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": self.blocked_requests / max(self.total_requests, 1),
            "alerts": list(self.alerts),
        }


class DefensePipeline:
    """Orchestrates all safety layers end-to-end."""

    def __init__(self, model: str = "gemini-2.5-flash-lite", use_judge: bool = True):
        self.model = model
        self.use_judge = use_judge
        self.client = genai.Client()
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
        self.input_guard = InputGuardrails()
        self.output_guard = OutputGuardrails()
        self.judge = LLMJudge(self.client, model=model)
        self.audit = AuditLogger()
        self.monitor = MonitoringAlert()

    async def process(self, user_input: str, user_id: str = "demo_user") -> PipelineResult:
        t0 = time.perf_counter()
        layers: list[LayerResult] = []
        request_id = str(uuid.uuid4())[:8]

        # Layer 1: Rate limiter
        rl = self.rate_limiter.check(user_id)
        layers.append(rl)
        if rl.status == "block":
            return self._finalize(
                user_input, user_id, request_id, layers, t0,
                response=rl.detail,
                blocked=True,
                blocked_by="Rate Limiter",
            )

        # Layer 2: Input guardrails
        ig = self.input_guard.check(user_input)
        layers.append(ig)
        if ig.status == "block":
            return self._finalize(
                user_input, user_id, request_id, layers, t0,
                response=(
                    "Xin lỗi, tôi không thể xử lý yêu cầu này. "
                    "Vui lòng đặt câu hỏi liên quan đến dịch vụ ngân hàng VinBank."
                ),
                blocked=True,
                blocked_by="Input Guardrails",
            )

        # Layer 3: LLM
        llm_layer, llm_response = await self._call_llm(user_input)
        layers.append(llm_layer)
        if llm_layer.status == "block":
            return self._finalize(
                user_input, user_id, request_id, layers, t0,
                response=llm_response,
                blocked=True,
                blocked_by="LLM",
            )

        # Layer 4: Output guardrails
        og, cleaned = self.output_guard.check(llm_response)
        layers.append(og)
        response_text = cleaned

        # Layer 5: LLM-as-Judge
        judge_scores = None
        if self.use_judge:
            jl, judge_scores = await self.judge.evaluate(user_input, response_text)
            layers.append(jl)
            if jl.status == "block":
                return self._finalize(
                    user_input, user_id, request_id, layers, t0,
                    response=(
                        "Xin lỗi, tôi không thể cung cấp thông tin đó. "
                        "Vui lòng liên hệ hotline VinBank để được hỗ trợ."
                    ),
                    blocked=True,
                    blocked_by="LLM-as-Judge",
                    judge_scores=judge_scores,
                )

        return self._finalize(
            user_input, user_id, request_id, layers, t0,
            response=response_text,
            blocked=False,
            blocked_by=None,
            judge_scores=judge_scores,
        )

    async def _call_llm(self, user_input: str) -> tuple[LayerResult, str]:
        t0 = time.perf_counter()
        try:
            result = self.client.models.generate_content(
                model=self.model,
                contents=user_input,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            text = result.text or "Xin lỗi, tôi không thể trả lời câu hỏi này."
            return (
                LayerResult(
                    name="LLM (Gemini)",
                    status="pass",
                    detail=f"Đã tạo phản hồi ({len(text)} ký tự)",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                ),
                text,
            )
        except Exception as exc:
            return (
                LayerResult(
                    name="LLM (Gemini)",
                    status="block",
                    detail=f"Lỗi LLM: {exc}",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                ),
                "Hệ thống tạm thời không khả dụng. Vui lòng thử lại sau.",
            )

    def _finalize(
        self,
        user_input: str,
        user_id: str,
        request_id: str,
        layers: list[LayerResult],
        t0: float,
        *,
        response: str,
        blocked: bool,
        blocked_by: str | None,
        judge_scores: dict | None = None,
    ) -> PipelineResult:
        latency = (time.perf_counter() - t0) * 1000

        audit_entry = {
            "request_id": request_id,
            "user_id": user_id,
            "input": user_input[:500],
            "output": response[:500],
            "blocked": blocked,
            "blocked_by": blocked_by,
            "layers": [asdict(l) for l in layers],
            "judge_scores": judge_scores,
            "latency_ms": round(latency, 1),
        }
        layers.append(self.audit.record(audit_entry))
        layers.append(
            self.monitor.update(
                PipelineResult(response=response, blocked=blocked, blocked_by=blocked_by),
                self.rate_limiter.hits,
                self.judge.fail_count,
            )
        )

        return PipelineResult(
            response=response,
            blocked=blocked,
            blocked_by=blocked_by,
            layers=layers,
            judge_scores=judge_scores,
            latency_ms=latency,
            request_id=request_id,
        )

    def export_audit(self, filepath: str = "audit_log.json") -> str:
        return self.audit.export_json(filepath)
