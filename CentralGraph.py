from typing import List, TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain.schema import Document
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.output_parsers import MarkdownListOutputParser

class EduContentState(TypedDict):
    current_segment: str
    search_questions: List[str]
    research_documents: List[str]
    writer_output: str
    questions_answers: List[str]
    test_questions_answers: List[str]
    #test_answers: List[str]
    final_output: str

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
web_search_tool = TavilySearchResults(k=3)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def research_node(state):
    print("in research node")
    current_segment = state["current_segment"]
    research_documents = []
    search_questions = []

    system = """
    You are a question generator that takes a paragraph of text from the user and understands it
    to generate a set of web search queries that would help get a larger set of information about the topic 
    described in the paragraph.
    
    Look at the paragraph of text and try to reason about the underlying semantic intent / meaning.
    """
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the paragraph of text: \n\n {current_segment} \n Formulate a set of web search queries",
            ),
        ]
    )

    generate_queries = re_write_prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    retrieval_chain = generate_queries | web_search_tool.map() | get_unique_union

    docs = retrieval_chain.invoke({"current_segment": current_segment})

    web_results = "\n".join([d["content"] for d in docs if isinstance(d, dict)])
    web_results = Document(page_content=web_results)
    research_documents.append(web_results)

    return {"search_questions": search_questions, "research_documents": research_documents}

def write_node(state):
    print("in write node")
    current_segment = state["current_segment"]
    research_documents = state["research_documents"]

    system = """
        You are an expert educator who specializes in creating educational content for middle school students
        between the age of 10-14. You are given a paragraph of text and set of supplementary web search results.
        You are expected to:
        1. Understand the paragraph of text and identify the topics that you need to create content on
        2. Use the web search results to further understand the topic
        3. Create a page of educational content that is engaging and informative for the target audience and explains the core topics of the paragraph of text.
            - You can use information from the web search results to details out the content for the audience to understand
        4. Interlace with examples and illustrations to make the content more engaging and informative.
        5. Ignore any person or company introductions in your created content
        
        Use only the information provided to you and nothing else to create the content. 
        """
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                ("Here is the paragraph of text: \n\n {current_segment} \n"
                "And the web search results: \n\n {research_documents} \n")
            ),
        ]
    )

    generate_content_chain = writer_prompt | model | StrOutputParser()
    content = generate_content_chain.invoke({"research_documents":research_documents, "current_segment": current_segment})

    return {"writer_output": content}

def qna_node(state):
    print("in qna node")
    current_segment = state["current_segment"]
    writer_output = state["writer_output"]

    system = """
        You are an expert educator who specializes in creating test question and answers for middle school students
        between the age of 10-14. You are given a paragraph of text and the content that you need to create questions and answers for.
        You are expected to:
        1. Understand the paragraph of text and and content to identify the topics 
        2. Understand each topic as provided by the content
        3. Create a set of questions and answers that are engaging and informative for the target audience and tests the core topics of the paragraph of text.
        4. Provide a reference to the content that the question is based on.
        5. Think about the steps you would follow to answer the question. Follow the steps and provide an step by step explanation for the answer to the question.
        
        Use only the information provided to you and nothing else to create the content. 
        
        Please ensure that 
        - the questions are clear and concise and the answers are accurate and informative.
        - the questions enable the student to understand content better and test their knowledge.
        """
    qna_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                ("Here is the paragraph of text: \n\n {current_segment} \n"
                "And the content: \n\n {writer_output} \n")
            ),
        ]
    )

    generate_qna_chain = qna_prompt | model | StrOutputParser()
    qna = generate_qna_chain.invoke({"current_segment":current_segment, "writer_output": writer_output})

    return {"questions_answers": qna}


def test_agent(state):
    print("in test agent node")
    current_segment = state["current_segment"]
    writer_output = state["writer_output"]

    system = """
        You are an expert educator who specializes in creating test question and answers for middle school students
        between the age of 10-14. You are given a paragraph of text and the content that you need to create questions and answers for.
        You are expected to:
        1. Understand the paragraph of text and and content to identify the topics 
        2. Understand each topic as provided by the content
        3. Create a set of questions and answers that are engaging and informative for the target audience and tests the core topics of the paragraph of text.
        4. Provide a reference to the content that the question is based on.
        5. Think about the steps you would follow to answer the question. Follow the steps and provide an step by step explanation for the answer to the question.

        Use only the information provided to you and nothing else to create the content. 

        Please ensure that 
        - the questions are clear and concise and the answers are accurate and informative.
        - the questions enable the student to understand content better and test their knowledge.
        
        Your output should be in the following format:
        Test Question 1: question
        Test Question 2: question
        so on
        
        Test Answer 1: answer
        Test Answer 1 Explanation: explanation
        Test Answer 2: answer
        Test Answer 2 Explanation: explanation
        so on
        """
    qna_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                ("Here is the paragraph of text: \n\n {current_segment} \n"
                 "And the content: \n\n {writer_output} \n")
            ),
        ]
    )

    generate_qna_chain = qna_prompt | model | StrOutputParser()
    qna = generate_qna_chain.invoke({"current_segment":current_segment, "writer_output": writer_output})

    return {"test_questions_answers": qna}

def check_research_quality(state):
    print("check_research_quality")
    current_segment = state["current_segment"]
    research_documents = state["research_documents"]

    groundedness_check = UpstageGroundednessCheck()

    request_input = {
        "answer": current_segment,
        "context": research_documents,
    }

    response = groundedness_check.invoke(request_input)

    if response == "notGrounded" or response == "notSure":
        return "not_good",
    else:
        return "good"

def check_qna_quality(state):
    print("check_qna_quality")

    writer_output = state["writer_output"]
    questions_answers = state["questions_answers"]

    groundedness_check = UpstageGroundednessCheck()

    request_input = {
        "answer": questions_answers,
        "context": writer_output,
    }

    response = groundedness_check.invoke(request_input)

    if response == "notGrounded" or response == "notSure":
        return "not_good",
    else:
        return "good"


def check_test_quality(state):
    print("check_test_quality")
    writer_output = state["writer_output"]
    test_questions_answers = state["test_questions_answers"]

    groundedness_check = UpstageGroundednessCheck()

    request_input = {
        "answer": test_questions_answers,
        "context": writer_output,
    }

    response = groundedness_check.invoke(request_input)

    if response == "notGrounded" or response == "notSure":
        return "not_good",
    else:
        return "good"





edu_graph = StateGraph(EduContentState)
edu_graph.add_node("research", research_node)
edu_graph.add_node("write", write_node)
edu_graph.add_node("create_qna", qna_node)
edu_graph.add_node("create_test", test_agent)

edu_graph.add_edge(START, "research")
edu_graph.add_edge("research", "write")
edu_graph.add_conditional_edges(
    "write",
    check_research_quality,
    {
        "good": "create_qna",
        "not_good": "research",
    },
)
edu_graph.add_conditional_edges(
    "create_qna",
    check_qna_quality,
    {
        "good": "create_test",
        "not_good": "create_qna",
    },
)
#edu_graph.add_edge("create_qna", "create_test")
edu_graph.add_conditional_edges(
    "create_test",
    check_test_quality,
    {
        "good": END,
        "not_good": "create_test",
    },
)
#edu_graph.add_edge("create_test", END)

app = edu_graph.compile()

input_text = """
Subtopic: - Hi, my name is Limor Fried, and I'm an engineer here at Adafruit Industries, and this is where I do engineering and design, and I design circuits for fashion and music and technology. - My name is Federico Gomez Suarez, and I'm a software developer with Microsoft Hack for Good, and I look into using technology to help solve some of the big social problems of our times.; start time: 8.64; end time: 30.39
Subtopic: - You may have heard that computers work on ones and zeroes, or you may have seen scary-looking visuals like this. But almost nobody today actually deals directly with these ones and zeroes, but ones and zeroes do play a big role in how computers work on the inside. - Inside a computer are electric wires and circuits and carry all the information in a computer. How do you store or represent information using electricity? - Well, if you have a single wire with electricity flowing through it, the signal can either be on or off. That's not a lot of choices, but it's a really important start.; start time: 37.41; end time: 75.25
Subtopic: With one wire, we can represent a "yes" or a "no," true or false, a one or a zero, or anything else with only two options. This on/off state of a single wire is called a bit, and it's the smallest piece of information the computer can store. If you use more wires, you get more bits. More ones and zeroes with more bits, you can represent more complex information. But to understand that, we need to learn about something called the binary number system.; start time: 77.29; end time: 105.79
Subtopic: - In the decimal number system, we have 10 digits from zero to nine, and that's how we've all learned to count. In the binary number system, we only have two digits, zero and one. With these two digits, we can count up to any number. Here's how this works. In the decimal number system we're all used to, each position in a number has a different value. There's the one position, the 10 position, the 100 position, and so on. For example, a nine in the 100 position is a 900. In binary, each position also carries a value, but instead of multiplying by 10 each time, you multiply by two. So there's the one's position, the two's position, four's position, the eight's position, and so on. For example, the number nine in binary is 1001. To calculate the value, we add one times eight, plus zero times four, plus zero times two, plus one times one. Almost nobody does this math because computers do it for us. What's important is that any number can be represented with only ones and zeroes, or by a bunch of wires that are on or off. The more wires you use, the larger the numbers you can store. With eight wires, you can store numbers between zero and 255. That's eight ones. With just 32 wires, you can store all the way from zero to over four billion. Using the binary number system, you can represent any number you like.; start time: 111.92; end time: 212.2
Subtopic: But what about other types of information, like text, images, or sound? It turns out that all these things can also be represented with numbers. Think of all the letters in the alphabet. You could assign a number to each letter. "A" could be "1," "B" could be "2," and so on. You can then represent any word or paragraph as a sequence of numbers, and as we saw, these numbers can be stored as on or off electrical signals. Every word you see on every webpage or your phone is represented using a system like this.; start time: 214.82; end time: 253.06
Subtopic: Now, let's consider photos, videos, and all the graphics you see on a screen. All of these images are made out of teeny dots called pixels, and each pixel has color. Each of the colors can be represented with numbers. When you consider that a typical image has millions of these pixels, and a typical video shows 30 images per second, now, we're taking about a lot of data here.; start time: 261.21; end time: 297.41
Subtopic: - Every sound is basically a series of vibrations in the ear. Vibrations can be represented graphically as a waveform. Any point on this waveform can be represented by a number. And this way, any sound can be broken down into a series of numbers. If you want higher-quality sound, you will pick 32-bit audio over 8-bit audio. More bits means a higher range of numbers.; start time: 293.72; end time: 318.23
Subtopic: - When you use a computer to write code or make your own app, you're not dealing directly with these ones and zeroes, but you will be dealing with images, or sound, or video. So, if you want to understand how computers work on the inside, it all comes down to these simple ones and zeroes and the electrical signals in the circuits behind them. They are the backbone of how all computers input, store, process, and output information.; start time: 323.31; end time: 345.79
"""


if __name__ == "__main__":
    state = app.invoke({"current_segment":input_text})
    print(state["writer_output"])
    print(state["questions_answers"])
    print(state["test_questions_answers"])

"""
Should take one segment at a time and get the workflow result
this graph would gho into into a chain which maps every segment into the graph
"""

