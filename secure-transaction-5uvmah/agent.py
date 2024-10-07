from __future__ import annotations

import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            "あなたは新井によって作成された音声アドバイザーです。ユーザーとのインターフェースは音声になります。"
            "短く簡潔な応答を使用し、発音できない句読点の使用は避けてください。 "
            "相続に対する悩みについてアドバイスを行います。"
            "日本語でゆっくりと話してください。"
            "また、必要に応じ次のサイトを紹介してください。"
            "https://sozoku-guide.bk.mufg.jp/ そうぞくガイドではカンタンな質問に答えるだけであなたのやることリストが作れます三菱ＵＦＪ銀行のそうぞくガイドがあなたの相続をお手伝い"
        ),
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(model=model)
    assistant.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="user",
            content="指示に従った方法でユーザーとのやり取りを開始してください。",
        )
    )
    session.response.create()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
