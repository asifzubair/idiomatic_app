# idiomatic_app

![idiomatic](idiomatic.jpeg)

AI agent makes you idiomatic

Many non-native English speakers — even those with strong proficiency — sometimes struggle to understand or confidently use idiomatic expressions. Idioms, whether common or obscure, can become barriers to communication when misunderstood, yet avoiding them can strip language of its natural rhythm and charm.

**Idiomatic** takes an active learning approach to help learners grasp and apply idioms in everyday conversation. Built with an agentic workflow and powered by spaced repetition, the tool delivers an engaging, personalized learning experience that grows with the user. While intended for non-native speakers, the tool can be used by anyone who wants to improve their language prowess, as it attempts to adapt to the user's level of expertise.

A few points of technicality here:
 1. I first came upon this issue when a fellow classmate, who was well-spoken, lamented how preparations for his [GMAT](https://www.mba.com/exams/gmat-exam) were severely impeded by idioms/expressions and he found them not intuitive.
 2. Previously, building such a tool might require constructing a database of known idioms, extensive classification etc. However, availability of powerful models like Gemini now allows one to deliver the tool based exclusively on smart prompting and well defined user workflows.

The tool relies on following Gen AI capabilities:
 1. Structured output/JSON mode/controlled generation
 2. Few-shot prompting
 3. Agents

We instantiate two LLMs for use in the workflow:
- an orchestration llm to function as a chat model
- a client model to generate output (using generate_content)

Our reasoning is simple. We want to separate the functioning of the two models so that one is more diverse in its output (using different temperature settings and prompts etc.) for question and answer generation. Where as the orchestration LLM is set to a lower temperture to have more predictable output.

## Question & Answer Generation
Since, this is the _meat and potatoes_ (ha!) of the app, we use a separate call to an LLM model to generate questions here. In this manner, I can adjust model parameters and as well have a more appropreiately engineered prompt. Currently, I'm using Gemini Flash right now, but we could also change this to a more sophisticated model while restircting output tokens thereby managing cost but elevating quality.

## Tools
I added tools for code interruptions like quitting, explanations or current score. Admittedly, adding this caused the app to become a bit brittle because my understanding of how tools would work was different from how they actually work in `LangChain`. Regardless, with some help from Gemini, I was able to hook up the tools such that reporting score and quitting work well. However, explanations are still a bit of hit or a miss.

I used the technique taught in the course of:

- listing the tools
- adding a ToolNode
- binding the tools to the LLM

## Orchestration LLM & Routing
I followed the Agents code lab to have a Chatbot node that could orchestrate different actions. I needed to refine the system prompt to strongly emphasize tool usage, otherwise tools were not being called. I also needed some help from Gemini, as I couldn't get the routing correct by myself and the code labs don't go deep into orchestration.

## The Agent Graph
The graph below shows the workflow. I had to add the get_input node as an intermediate as I wanted the user to break the question and answer flow to interact with the tutor at any instance. Presently, the chatbot engages in some niceties, generates questions for the user to answer and then scores them. At any instance the user might provide a natural language prompt, at which point the Chatbot will either use its tools to reply or politely that it doesn't know, at which point the user can continue playing the quiz.

## Final Remarks
This was a very fun exercise for me from which I learned a lot of new things. I personally used the tool myself and found a few idioms that I had been using wrongly in the past. I left in the debug statements, as I think it is cool to see how routing is happening. Despite some bias, I do think this is a cool app, however, the present implementation has some shortcomings that I must disclose:

- the implementation is a brittle and will not scale well
- while I tried to tweak model parameters for diverse output, I still see the same idioms being repeated several times
- when asked to explain the last idiom, or why user got the answer wrong, the chatbot will often default to the idiom "break a leg" and not explain the idiom that is being studied
- data persistence doesn't work well, so while the app quits graefully, it is not able to save user data
- another consequence of not saving user data is that spaced repetition couldn't be implemented properly, it's something I am keen to revisit later on

Having said, the chat interface does allow the user to correct the bot and when prodded the chatbot politely informs the user what might be going wrong with the workflow. Redemption, yet ?