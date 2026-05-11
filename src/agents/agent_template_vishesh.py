This is how we can create custom agents and one must add agebt name in registry _REAL

# from ..llm import LLMClient

# class AgentName:
#     name = "real name of agent"
#     async def run(
#         self,
#         *,
#         query:str,
#         user_context:dict[str,Any],
#         classification:dict[str,Any],
#         llm:LLMClient | None=None
#     ) -> AsyncIterator[dict[str,Any]]:
#     # IMPL
#     yield {"type": "data", "delta":"I am your agent..."}

#     # IMPL
#     yield {"type": "structured", "payload":{
#         "agent":self.name,
#         ...
#     },
#     }
