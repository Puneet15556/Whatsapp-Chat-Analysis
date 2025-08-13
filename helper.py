

import os
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = "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"





from langchain.prompts import PromptTemplate

from urlextract import URLExtract






extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df


def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def chat_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # Only use the last 20 messages
    chat_messages = "\n".join(df["message"].astype(str).tail(50))
    
    prompt = PromptTemplate(
        input_variables=["chat_messages"],
        template="""
        You are a helpful assistant that analyzes chat messages.
        Give the Summary of the chat messages.
        {chat_messages}
        """
    )
    
  
      
    
    
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,)
    chain = prompt | llm
    response = chain.invoke({"chat_messages": chat_messages})
    return response.content     
        
 

    














