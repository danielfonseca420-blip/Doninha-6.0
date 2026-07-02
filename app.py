import chainlit as cl
import ollama

# 
# Configuraes fixas (mude aqui o que precisar)
# 
MODEL_NAME = "doninha8:latest"          #  modelo local Doninha8 para o middleware
SYSTEM_PROMPT = """
Voc  um assistente extremamente til, direto e sarcstico quando faz sentido.
Responda em portugus do Brasil, de forma clara e concisa.
"""  # personalize bastante aqui!cd "D:\Desktop\IA Doninha\Ollama-MMLU-Pro"

# 

@cl.on_chat_start
async def start():
    # Mensagem de boas-vindas que aparece quando o usurio entra
    await cl.Message(
        content="Bem Vindo  Inteligncia Artificial da Operao Doninha! No faa perguntas idiotas"
    ).send()

    # Opcional: mostra um "pensando..." enquanto carrega
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    # Pega o histrico da sesso (para manter contexto)
    history = cl.user_session.get("history") or []

    # Adiciona a mensagem do usurio no histrico
    history.append({"role": "user", "content": message.content})

    # Mostra "pensando..." na interface
    msg = cl.Message(content="")
    await msg.send()

    # Chama o Ollama com streaming (resposta aparece letra por letra)
    try:
        stream = ollama.chat(
            model= "gpt-oss:120b-cloud",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history
            ],
            stream=True,
            think="high",   # ← String: "low", "medium" ou "high",
            options={
                "temperature": 0.7,
                "num_ctx": 8192  # aumenta se seu modelo suportar mais contexto
            }
        )

        full_response = ""

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                full_response += token
                await msg.stream_token(token)

        # Finaliza a mensagem
        await msg.update()

        # Salva a resposta da IA no histrico
        history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("history", history)

    except Exception as e:
        await cl.Message(
            content=f"Ops... deu ruim aqui: {str(e)}\nTenta de novo?"
        ).send()


# Opcional: boto para limpar conversa
@cl.action_callback(name="Limpar conversa")
async def clear_conversation():
    cl.user_session.set("history", [])
    await cl.Message(content="Conversa zerada! Pode comear do zero.").send()
