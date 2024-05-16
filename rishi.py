import os
import tempfile
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.environ.get("OPENAI_API_KEY")),
        file_path, 
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        # Check if the user's message contains keywords related to different types of plots
        if "scatter plot" in user_message.lower():
            # Generate scatter plot directly
            df = pd.read_csv(file_path)
            x_column, y_column = extract_plot_columns(user_message)
            if x_column is not None and y_column is not None and x_column in df.columns and y_column in df.columns:
                plt.scatter(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title('Scatter Plot')
                # Save the plot to a temporary file
                plt.savefig("scatter_plot.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "scatter_plot.png"}
            else:
                return {"error": "Unable to identify appropriate columns for scatter plot."}
        elif "bar chart" in user_message.lower():
            # Generate bar chart directly
            df = pd.read_csv(file_path)
            x_column, y_column = extract_plot_columns(user_message)
            if x_column is not None and y_column is not None and x_column in df.columns and y_column in df.columns:
                plt.bar(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title('Bar Chart')
                # Save the plot to a temporary file
                plt.savefig("bar_chart.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "bar_chart.png"}
            else:
                return {"error": "Unable to identify appropriate columns for bar chart."}
        elif "line plot" in user_message.lower():
            # Generate line plot directly
            df = pd.read_csv(file_path)
            x_column, y_column = extract_plot_columns(user_message)
            if x_column is not None and y_column is not None and x_column in df.columns and y_column in df.columns:
                plt.plot(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title('Line Plot')
                # Save the plot to a temporary file
                plt.savefig("line_plot.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "line_plot.png"}
            else:
                return {"error": "Unable to identify appropriate columns for line plot."}
        elif "box plot" in user_message.lower():
            # Generate box plot directly
            df = pd.read_csv(file_path)
            column = extract_plot_column(user_message)
            if column is not None and column in df.columns:
                sns.boxplot(x=df[column])
                plt.xlabel(column)
                plt.title('Box Plot')
                # Save the plot to a temporary file
                plt.savefig("box_plot.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "box_plot.png"}
            else:
                return {"error": "Unable to identify appropriate column for box plot."}
        elif "line graph" in user_message.lower() or "line plot" in user_message.lower():
            # Generate line graph directly
            df = pd.read_csv(file_path)
            x_column, y_column = extract_plot_columns(user_message)
            if x_column is not None and y_column is not None and x_column in df.columns and y_column in df.columns:
                plt.plot(df[x_column], df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title('Line Graph')
                # Save the plot to a temporary file
                plt.savefig("line_graph.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "line_graph.png"}
            else:
                return {"error": "Unable to identify appropriate columns for line graph."}
        # If the query is not related to visualization, pass it to the CSV agent
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        
        response = agent.run(tool_input)

        # Extract Python code from the response if available
        code = extract_code_from_response(response)
        if code:
            # Check if the response contains a plot command
            if "plt.plot" in code or "sns.scatterplot" in code:
                # Execute the code to generate the plot
                exec(code)
                # Save the plot to a temporary file
                plt.savefig("plot.png")
                # Close the plot to avoid displaying it in the Streamlit app
                plt.close()
                # Return the path to the saved plot
                return {"plot": "plot.png"}
        
        # If the response does not contain a plot command, return the response as is
        return response
        
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def extract_plot_columns(user_message):
    """Extracts columns for plot from user's message."""
    # Parse user's message to extract column names
    # Example: "Show a scatter plot between GDP per capita and Healthy life expectancy."
    columns = re.findall(r"'(.*?)'", user_message)
    if len(columns) == 2:
        return columns[0], columns[1]  # Return X and Y columns
    else:
        return None, None  # Return None if unable to extract columns

def extract_plot_column(user_message):
    """Extracts column for histogram plot from user's message."""
    # Parse user's message to extract column name
    # Example: "Display a histogram of the distribution of Score."
    columns = re.findall(r"of\s(.*?)\.", user_message)
    if columns:
        return columns[0]  # Return the column name
    else:
        return None  # Return None if unable to extract column
def extract_code_from_response(response):
    """Extract Python code from the response."""
    if "tool_output" in response and "stdout" in response["tool_output"]:
        output = response["tool_output"]["stdout"]
        code_blocks = re.findall(r"```(.*?)```", output, re.DOTALL)
        for code_block in code_blocks:
            if any(keyword in code_block for keyword in ["plt.plot", "sns.scatterplot"]):
                return code_block
    return None


def perform_statistical_analysis(df, user_message):
    """Perform statistical analysis on the DataFrame based on the user's query."""
    # Define keywords for statistical analysis
    keywords = {
        'mean': 'mean',
        'median': 'median',
        'mode': 'mode',
        'standard deviation': 'std',
        'correlation': 'corr',
    }
    
    # Find the keyword in the user's message and perform corresponding statistical analysis
    for keyword, function in keywords.items():
        if keyword in user_message.lower():
            if keyword == 'correlation':
                # Extract column names for correlation calculation
                columns = re.findall(r"'(.*?)'", user_message)
                if len(columns) == 2:
                    return df[columns[0]].corr(df[columns[1]])
                else:
                    return {"error": "Unable to extract column names for correlation calculation."}
            elif keyword == 'mean':
                # Check if the query is asking for the mean value of a specific column
                column = re.findall(r"of\s'(.*?)'", user_message)
                if column:
                    return df[column[0]].mean()
                else:
                    return {"error": "Unable to extract column name for mean calculation."}
            elif keyword == 'median':
                # Check if the query is asking for the median value of a specific column
                column = re.findall(r"of\s'(.*?)'", user_message)
                if column:
                    return df[column[0]].median()
                else:
                    return {"error": "Unable to extract column name for median calculation."}
            elif keyword == 'mode':
                # Check if the query is asking for the mode value of a specific column
                column = re.findall(r"of\s'(.*?)'", user_message)
                if column:
                    return df[column[0]].mode()[0]  # Mode may have multiple values, return the first one
                else:
                    return {"error": "Unable to extract column name for mode calculation."}
            elif keyword == 'standard deviation':
                # Check if the query is asking for the standard deviation of a specific column
                column = re.findall(r"of\s'(.*?)'", user_message)
                if column:
                    return df[column[0]].std()
                else:
                    return {"error": "Unable to extract column name for standard deviation calculation."}
    
    return {"error": "Statistical analysis not performed."}

def generate_plot(df, user_message):
    """Generate plots based on the user's query."""
    if "scatter plot" in user_message.lower():
        # Generate scatter plot
        columns = re.findall(r"'(.*?)'", user_message)
        if len(columns) == 2:
            plt.scatter(df[columns[0]], df[columns[1]])
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title('Scatter Plot')
            plt.savefig("scatter_plot.png")
            plt.close()
            return {"plot": "scatter_plot.png"}
        else:
            return {"error": "Unable to extract column names for scatter plot."}
     
    elif "bar chart" in user_message.lower():
        # Generate bar chart
        columns = re.findall(r"'(.*?)'", user_message)
        if len(columns) == 2:
            plt.bar(df[columns[0]], df[columns[1]])
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title('Bar Chart')
            plt.savefig("bar_chart.png")
            plt.close()
            return {"plot": "bar_chart.png"}
        else:
            return {"error": "Unable to extract column names for bar chart."}

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""
    st.title('Chat with CSV using llama 2 🦙')
    st.write('Please upload your CSV file and enter your query below:')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,
                        "FileSize": uploaded_file.size}
        st.write(file_details)

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            file_path = temp_file.name

        df = pd.read_csv(file_path)
        st.dataframe(df)

        user_input = st.text_input("Your query")
        if st.button('Run'):
            response = csv_agent_func(file_path, user_input)
            st.write("Response from csv_agent_func:", response)  # Debug statement

            if response is not None:
                # Check if the response contains a plot
                if "plot" in response:
                    plot_url = response["plot"]
                    st.image(plot_url, caption='Plot', use_column_width=True)

                # Display other content from the JSON response
                # You can add more conditions here to handle different types of responses
                else:
                    st.write(response)

    st.divider()
csv_analyzer_app()