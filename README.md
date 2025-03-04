# AI Course Development Platform

An intelligent Streamlit application that leverages AI to automatically generate comprehensive course curricula and detailed lesson plans on any subject.

## Overview

This application uses CrewAI's Flow framework and the Gemini 1.5 Flash model to create structured educational content with minimal user input. The system enhances course materials by incorporating real-time data from news sources and academic papers related to the chosen subject.

## Features

- **AI-Generated Course Outlines**: Creates structured course outlines with customizable module count
- **Detailed Lesson Plans**: Generates comprehensive lesson plans for each module
- **Real-World Context Integration**: Incorporates current news and academic research 
- **Multiple Export Formats**: Export courses in JSON, Markdown, or PDF formats
- **Interactive User Selection**: Choose specific modules for detailed lesson plan generation


## Dependencies

This project requires the following Python packages:
- streamlit
- crewai
- litellm
- requests
- fpdf
- json
- re

## Setup and Configuration

1. Ensure you have UV installed for project management
2. Install the required dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Configure your API keys:
   - Replace `"api_key_here"` in the News API section
   - Replace `"api_key"` in the LiteLLM completion calls

## Running the Application

```bash
streamlit run main.py
```

Navigate to the URL displayed in your terminal (typically http://localhost:8501).

## Usage Guide

1. Enter a course subject or theme (e.g., "Machine Learning", "Digital Marketing")
2. Specify the desired number of modules (1-20)
3. Click "Generate Course"
4. The system will:
   - Fetch relevant news and research papers
   - Generate a course outline
   - Allow you to select modules for detailed lesson plans
5. Review the generated content and download in your preferred format (JSON, MD, PDF)

## How It Works

The application uses a structured flow:

1. **Subject Input**: Collects the course topic and module count
2. **External Data Collection**: Fetches relevant news and academic papers using APIs
3. **Course Outline Generation**: Uses Gemini 1.5 Flash to create a structured outline
4. **Lesson Plan Creation**: Generates detailed content for each selected module
5. **Export Options**: Provides multiple download formats for the final course

## Customization

- Change the AI model by modifying the `model` attribute in `CourseDevelopmentFlow`
- Add additional data sources in the `fetch_external_data` method
- Customize PDF formatting in the `generate_pdf_content` function

## Notes

- This application requires active internet connection for API calls
- API rate limits may apply to the News API and Semantic Scholar API
- For large courses, generation may take several minutes

## License

This project is licensed under the MIT License.