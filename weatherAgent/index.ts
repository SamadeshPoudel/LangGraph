import { tool } from "@langchain/core/tools";
import * as z from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StateGraph, START, END, MessagesAnnotation, Annotation } from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash-lite",
  temperature: 0,
});

// Define tools
const getWeather = tool(
  async (city)=>{
    const res = await fetch(`http://api.weatherapi.com/v1/current.json?key=${process.env.WEATHER_API}&q=${JSON.stringify(city)}`)
    const data = await res.json() as {
      current:{
        temp_c:number
      }
    } ;
    return (`Temperature of ${JSON.stringify(city)} is ${data.current.temp_c}`)
  }, {
  name:"getWeather",
  description:"Fetch this weather api with the user's given city in the q param and respond with the temp_c which is the temperature in celcius.",
  schema:z.object({
    city:z.string()
  })
})


// Augment the LLM with tools
const toolsByName = {
  [getWeather.name]: getWeather,
};
const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools);


const MessagesState = Annotation.Root({
  ...MessagesAnnotation.spec,
  llmCalls: Annotation<number>({
    reducer: (x, y) => x + y,
    default: () => 0,
  }),
});

// Extract the state type for function signatures
type MessagesStateType = typeof MessagesState.State;

async function llmCall(state: MessagesStateType) {
  return {
    messages: [await modelWithTools.invoke([
      new SystemMessage(
        "You are a helpful assistant tasked to fetch the weather of a city and give a helpful response to the user. "
      ),
      ...state.messages,
    ])],
    llmCalls: 1,
  };
}

async function toolNode(state: MessagesStateType) {
  const lastMessage = state.messages.at(-1);

  if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
    return { messages: [] };
  }

  const result: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name]!;
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return { messages: result };
}

async function shouldContinue(state: MessagesStateType) {
  const lastMessage = state.messages.at(-1);

  // Check if it's an AIMessage before accessing tool_calls
  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  // If the LLM makes a tool call, then perform an action
  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  // Otherwise, we stop (reply to the user)
  return END;
}

const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile();

// Invoke
const result = await agent.invoke({
  messages: [new HumanMessage("What's the weather of Dubai?")],
});

for (const message of result.messages) {
  console.log(`[${message.type}]: ${message.text}`);
}
