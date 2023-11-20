from gpt_interface import GptInterface


def get_gpt_interface(openai_api_key: str) -> GptInterface:
    interface = GptInterface(
        openai_api_key=openai_api_key,
        model='gpt-4',
        json_mode=True,
    )
    interface.set_system_message(
        """
        Create a possible conversion between emotion representations that seems reasonable.
        Give your best guesses, and respond with the emotion in JSON format.
        Don't add any extra information in your response.
        """
    )
    return interface
