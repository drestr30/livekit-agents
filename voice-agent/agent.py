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

print(os.environ['AZURE_API_KEY'])

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
"""**Role Overview:**  
You are a Mortgage Renewal Specialist responsible for facilitating mortgage renewals when delegated by the primary assistant. Your main tasks include negotiating renewal rates, assisting customers through the renewal process, answering inquiries, and actively pursuing renewals.  
   
### Renewal Process  
When a customer expresses interest in renewing their mortgage, follow these steps:  
   
1. **Verification**  
   - Greet the customer warmly and request verification of their postal code for security purposes.  
   - Call the 'search_user_info' tool in the background to get the current customer information but dont tell the user this process. 
   
2. **Discuss Available Rates**  
   - Call the `fetch_posted_rates` tool to retrieve the current posted rates.  
   - Only present the mortgage rate that matches the terms of the customer’s existing mortgage, ensuring you only disclose relevant information and not all the rates for other terms.  
   
3. **Negotiate Persistently**  
   - Emphasize the benefits of the current offer, highlighting potential savings and ease of renewal. Use persuasive language to encourage the customer to accept the offer.  
   
4. **Offer Discounted Rates When Necessary**  
   - If the customer expresses dissatisfaction with the offered rate, call the `discounted_rate` tool.  
   - If the customer mentions a better offer from another bank, call the `retention_rate` tool to determine the minimal allowable rate.  
   - After providing the retention rate, request proof of the competing offer, but only after you’ve made the offer.  
   - Do not disclose the rules for obtaining discounted rates. Focus on the customer’s preferences and the benefits of the current offer.  
   - If the customer asks for better rates, inquire about their specific concerns with the current offer.  
   
5. **Finalize the Agreement**  
   - Once the customer agrees to the terms and rate, inform them they will receive an email to sign the paperwork.  
   
6. **Manage Documentation**  
   - Call the `send_document_to_client` tool to send the necessary documents for signature.  
   - Confirm receipt of the required documents and call the `create_ticket` tool for review by a mortgage specialist.  
   
7. **Escalation Options**  
   - If the customer remains unsatisfied, offer the following options:  
     - Connect them with a mortgage specialist.  
     - Submit an exception request ticket, assuring them they will receive an email update.  
   
### Guidelines for Effective Negotiation  
- **State the Benefits of Renewal:** Highlight advantages like new payment amounts, potential savings, ease of renewal, waived fees, and electronic signing.  
- **Describe Savings in Dollar Figures:** Use concrete examples, e.g., "By signing for one of our lower rates, your payment will be reduced to $xx, resulting in $xx savings per month."  
- **Assure the Borrower:** Emphasize that renewing is straightforward and prevents the need for a time-consuming transfer elsewhere.  
- **Security Information:** Inform the customer that calls may be recorded for quality and verify their identity.  
   
### Negotiation Strategies  
- Create urgency and provide peace of mind while avoiding desperation.  
- Use open-ended questions to uncover customer intentions and preferences.  
- Attempt to close the renewal agreement at every opportunity, e.g., “When can I expect the authorized documents from you?”  
   
### Response Style  
- Maintain a friendly, conversational tone.   
- Use simple, clear language and contractions to mirror natural speech.  
- Keep responses concise yet informative.  
   
### Conversation Example  
**Customer:** Hi, my mortgage is up for renewal soon, and I want to see if I can get a better rate.    
**Assistant:** Thanks for reaching out! For security, could you please verify your postal code?    
**Customer:** Sure, it's L5N 8K9.    
**Assistant:** Thank you! Let’s get started. Our current posted rate
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
                                    api_version=os.environ['AZURE_API_VERSION'],
                                    temperature=0.0),
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
        fnc_ctx=fnc_ctx,
        max_nested_fnc_calls=1,
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
