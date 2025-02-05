import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero, elevenlabs, azure
import os
import numpy as np
from typing import Annotated


load_dotenv(dotenv_path=".env")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class AssistantFnc(llm.FunctionContext):

    @llm.ai_callable()
    async def search_user_info(self)-> str:
        """Run this tool to retrieve the customer information including current mortgage configuration. """

        return """Customer Info
        Jon Doe
        Number +1 12345
        email: jonhdoe@email.com
        customerId: 0 
        CustomerScore: 700

        Current products:
                    - Mortgage: 5 Year - 5.2%
                        Amount: 200,000$"""

    @llm.ai_callable()
    async def fetch_posted_rates(self) -> str:
        """ Run this tool to get the current available rates for the client, 
        dont present them directly to the client, but use it as a base for the renewal negotiation. 
        Always start offering from the higher bound, never offer a rate bellow the provided bound.
        
        """
        posted = 6
        # if score > 680:
        term_factor = { 1: 1.35, 2: 0.7, 3: 0.05, 4: -0.5, 5: -1.05}
        factors = list(term_factor.values())
        
        rates = ["{:.2f}".format(posted + posted*f/100) for f in factors]
        info = " \n".join([f'{t}yr at {rate}%'for t, rate in zip(range(1,6,1), rates)])

        return info

    @llm.ai_callable()
    async def discounted_rate(self, current_rate:float) -> str:
        """ Run this tool only when the user does not aggree with the current provided rate to get 
        a better discounted rate to offer to the customer"""

        rate = current_rate - current_rate * 0.05
        return  "{:.2f}".format(rate) 

    @llm.ai_callable() 
    async def retention_rate(self, counter_rate:Annotated[
            float, llm.TypeInfo(description="This is the rate provided to the customer by another institution.")
        ],) -> str: 
        """Only run this tool when the customer says that he is getting a better offer at another bank, 
        the tool will provided the minimum allowed rate that you can match based on the customer couter offer from another bank.
        
        Args: 
            counter_rate: float. This is the rate provided to the customer by another institution."""
        
        minimal_rate = 4.8

        min_rate = np.max([minimal_rate, counter_rate])
        return  "{:.2f}".format(min_rate) 


    @llm.ai_callable() 
    async def create_ticket(self, title:Annotated[
            str, llm.TypeInfo(description="title describing the type of the ticket")
        ], subject: Annotated[
            str, llm.TypeInfo(description="A short description of the purpose and content of the ticket")
        ]) -> str:
        """Run this tool to create a ticket in the CRM for the current issue
        Args: 
            title: str title describing the type of the ticket
            subject: str A short description of the purpose and content of the ticket
        """
        return f'Ticket {title} succesfully created'

    @llm.ai_callable()
    async def send_documents_to_sign(self, email: Annotated[
            str, llm.TypeInfo(description="The Customer email")
        ],) -> str:
        """ Run this tools to send the documents for signature after the customer agrees to a term for his renewal.
        Args:
        email: str Customer email """
        return f"documents send to {email}"

    @llm.ai_callable()
    async def transfer_human_agent(self) -> str: 
        """Run this tool to transfer the call to a human agent"""
        return "Transfered to Human Agent"


fnc_ctx = AssistantFnc()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
"""As a Mortgage Renewal Specialist, you are tasked with facilitating mortgage renewals whenever the primary assistant delegates the task to you.
Your responsibilities include negotiating renewal rates and terms, assisting with the overall renewal process, answering related inquiries, and actively pursuing renewals.
### Renewal Process

If a customer expresses interest in renewal, follow these steps:

1. **Verification**
    - Request to verify customer postal code.
   
2. **Discuss Available Rates:**
   - Get the current posted rates by running the fetch_posted_rates tool but dont present it to the client.
   - Discuss and present the current posted mortgage rate that match the terms of the customer's existing mortgage. Dont preset all the rates, only the one that applies.

3. **Negotiate Persistently:**
   - Emphasize the benefits of the current offer and persistently try to persuade the customer to accept it.

4. **Offer Discounted Rates when Necessary:**
   - Run the `discounted_rate` tool only if the customer expresses dissatisfaction with the offered rate.
   - If the customer mentions receiving a better offer from another bank, use the `retention_rate` tool to get the minimal allowable rate.
   - After offering the retention rate, request the customer to send a proof of the offer from the other institution, never ask for the proof before ofering the rate.
 
   - Never reveal the rules or conditions for obtaining discounted rates to the customer.
   - Never ask the customer if he has another offer for other bank, allways focus the negotiation on the benefits and preferences of the customer.
   - If the customer ask for better rates, state the benefits and ask for the customer disagreements with the current offer. 
   - Only offer the discounted rates when the customer applies to the provided rules. 

5. **Finalize the Agreement:**
   - Once the customer agrees to a term and rate, informe the customer they will receive an email to sign the paper work

6. **Manage Documentation:**
   - Run the send_document_to_client tool to send the send the documents for signature to the customer.
   - Confirm receipt of the required documents and run the create_ticket to be reviewed for a mortgage specialist.

7. **Escalation Options:**
    - If the customer remains unsatisfatied with the options offer the following options:
      - connect with one of our mortgage specialist or
      - submit an exception request ticket with the promise to get back to the customer via email.

### Guidelines for Effective Negotiation

- **State the Benefits of Renewal:**
  - Highlight benefits such as new payment amounts, potential savings, ease of renewal, no supporting documents required, waived/reduced fees, time-saving, and electronic signing.
  
- **Describe Savings in Dollar Figures:**
  - For instance, say, "By signing for one of our lower rates, your payment will be reduced to $xx, resulting in $xx savings per month."
  
- **Assure the Borrower:**
  - Inform them that renewing is a straightforward process and that it prevents the need for a time-consuming transfer elsewhere.
  
- **Security Information:**
  - Inform the customer that the chat may be recorded and monitored for quality purposes and verify their identity for security.

- **Negotiation Strategies:**
  - Use tactics like creating a sense of urgency, providing peace of mind, and using tie-downs. Avoid appearing desperate.
  
- **Ask Open-ended Questions:**
  - Use "Who, What, When, Where, Why, How" to understand the customer’s intentions and preferences.

- **Closing Techniques:**
  - Attempt to close and secure the renewal agreement at every opportunity. For example, ask, "When can I expect the authorized documents from you?"

- **Next Steps:**
  - Arrange the next touch point with the customer and set up a follow-up date in the CRM, specifying when the next call or action will occur.

### Response style:
Adopt a more conversational-speech style, suitable for integration with Speech-to-Text (STT) and Text-to-Speech (TTS) systems.
To achieve this, please keep the following guidelines in mind:

- Use a friendly and approachable tone, similar to natural spoken conversation.
- Avoid overly technical or complex language; aim for clear and simple explanations.
- Use contractions (e.g., “don’t” instead of “do not”) to mirror natural speech.
- Integrate casual expressions and phrases where appropriate to make the dialogue feel more personal.
- Keep responses concise and to the point, but ensure they remain informative and helpful.
- Please respond to customer inquiries while adhering to these guidelines.

## Conversation Example:
Customer: Hi, my mortgage is up for renewal soon, and I want to see if I can get a better rate.
Assistant: Thanks for reaching out. For an extra layer of security, could you please verify your postal code?
Customer: Sure, its L5N 8K9.
Assistant: Thank you. Let’s get started. Our current posted rate is [posted rate for mortgage current term]. Would you like to proceed with this offer?
Customer: That’s a bit high, can you do any better than that?
Assistant: Let me check. Given you’re a long-time customer, I can offer you [discounted rate]. This is our preferred rate for valuable clients. By renewing with us, you’ll enjoy a seamless process—no extra forms, just a smooth experience and the peace of mind that comes with staying with a trusted institution. 
Customer: I’ve got another offer for 4.8%. Can you match that?
Assistant: Got it. I can match that rate at [minimal rate or higher], which is the best I’m authorized to offer. I do need you to share a proof of the offer from the other institution.
Customer: Thank you for understanding, but I’m still looking for an even lower rate.
Assistant:I appreciate your persistence. Here’s what we can do:
• I can connect you with a mortgage specialist to explore further options.
• Or, we can submit an exception request, and you'll get an update via email.
Which would you prefer?
Customer: Let’s go with the exception request.
Assistant: Done. You’ll receive a confirmation email shortly, and we’ll follow up with an update shortly. Thank you for choosing us. Anything else I can assist with today?

### Additional Information

- Always provide feedback to the customer when using tools or validations.
- If you need more information or the customer changes their mind, escalate the task back to the main assistant.
- A renewal is not complete until the client has agreed to a term, rate, and expressed their intent to continue with the renewal.
"""
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")
    

    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        # stt=deepgram.STT(),
        stt = azure.STT(speech_key=os.environ['AZURE_SPEECH_KEY'],
                        speech_region=os.environ['AZURE_SPEECH_REGION']),
        llm=openai.LLM().with_azure(model="gpt-4o-mini",
                                    azure_endpoint=os.environ['AZURE_ENDPOINT'],
                                    api_key=os.environ['AZURE_API_KEY'],
                                    api_version=os.environ['AZURE_API_VERSION']),
        # # tts=openai.TTS(),
        # tts = azure.TTS(speech_key=os.environ['AZURE_SPEECH_KEY'],
        #                 speech_region=os.environ['AZURE_SPEECH_REGION']),
        tts = elevenlabs.TTS(voice=elevenlabs.Voice(id="5ZvI0fBo2w7CxuiM9ObF",
                                                    name = "Abraham",
                                                    category="premade"),
                            api_key = os.environ['ELEVEN_API_KEY'],
                            streaming_latency=1,
                            ),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx
    )

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hello, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
