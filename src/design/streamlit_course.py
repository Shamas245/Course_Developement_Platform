import streamlit as st
from crewai.flow.flow import Flow, start, listen
from litellm import completion
import requests
import json
from fpdf import FPDF
import re  # For filtering module headings

def generate_pdf_content(subject, lesson_plans):
    """
    Generate a PDF document from the course subject and lesson plans.
    Returns the PDF as a byte string.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"{subject} Course Curriculum", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for i, (module, plan) in enumerate(lesson_plans.items(), start=1):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Module {i}: {module}", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, plan)
        pdf.ln(5)
    pdf_bytes = pdf.output(dest="S").encode("utf-8")
    return pdf_bytes

class CourseDevelopmentFlow(Flow):
    # Use Gemini as the model for LLM generation.
    model = "gemini/gemini-1.5-flash"
    
    # Attribute to hold module count.
    module_count = 5

    @start()
    def get_subject(self):
        # These values will be passed in from Streamlit.
        return self.subject_input

    @listen(get_subject)
    def fetch_external_data(self, subject):
        with st.status("Fetching news data...", expanded=True) as status:
            news_api_key = "api_key_here"
            news_url = "https://newsapi.org/v2/everything"
            news_params = {
                "q": subject,
                "sortBy": "publishedAt",
                "apiKey": news_api_key,
                "pageSize": 5
            }
            try:
                news_response = requests.get(news_url, params=news_params)
                news_data = news_response.json()
                news_articles = [article["title"] for article in news_data.get("articles", [])]
                status.update(label="News data fetched successfully", state="complete")
                st.success("News data fetched successfully")
            except Exception as e:
                st.error(f"Error fetching news data: {e}")
                news_articles = []
            
            status.update(label="Fetching Semantic Scholar data...")
            scholar_papers = self.fetch_semantic_scholar_data(subject)
            return (news_articles, scholar_papers)

    def fetch_semantic_scholar_data(self, subject):
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": subject,
            "limit": 5,
            "fields": "title,authors,year,abstract,url"
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            papers = [paper["title"] for paper in data.get("data", [])]
            st.success("Semantic Scholar data fetched successfully")
            return papers
        except Exception as e:
            st.error(f"Error fetching Semantic Scholar data: {e}")
            return []

    @listen(fetch_external_data)
    def combine_external_context(self, external_data):
        news_articles, scholar_papers = external_data
        
        if news_articles:
            with st.expander("News Headlines"):
                for i, article in enumerate(news_articles, 1):
                    st.write(f"{i}. {article}")
        
        if scholar_papers:
            with st.expander("Research Papers"):
                for i, paper in enumerate(scholar_papers, 1):
                    st.write(f"{i}. {paper}")
        
        external_context = (
            f"News Headlines: {', '.join(news_articles)}. "
            f"Research Papers: {', '.join(scholar_papers)}."
        )
        return external_context

    @listen(combine_external_context)
    def generate_course_outline(self, external_context):
        with st.status("Generating course outline...", expanded=True) as status:
            prompt = (
                f"Using the following context from recent news and scholar papers trends:\n"
                f"{external_context}\n\n"
                f"Generate a comprehensive course outline on the subject '{self.subject_input}' with {self.module_count} modules. "
                "Each module should have a title and a brief description, formatted as a numbered list (e.g., '1. Module Title: Description')."
            )
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key="api_key"
            )
            outline = response["choices"][0]["message"]["content"].strip()
            status.update(label="Course outline generated", state="complete")
            return outline

    @listen(generate_course_outline)
    def generate_lesson_plans(self, outline):
        st.subheader("Course Outline")
        st.markdown(outline)
        
        # Split the outline into lines.
        lines = [line.strip() for line in outline.split("\n") if line.strip()]
        # Filter lines that match a numbered module format (e.g., "1. Module Title: Description").
        modules = [line for line in lines if re.match(r"^\d+\.", line)]
        if not modules:
            modules = lines
        
        # Let the user select which modules they want detailed lesson plans for.
        selected_modules = st.multiselect(
            "Select modules for detailed lesson plans:",
            modules,
            default=modules
        )
        
        lesson_plans = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, module in enumerate(selected_modules):
            status_text.text(f"Generating lesson plan for module {i+1}/{len(selected_modules)}")
            prompt = (
                f"Based on the course module: '{module}', generate a detailed lesson plan. "
                "Include key topics, activities, and suggested resources."
            )
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key="api_key"
            )
            lesson_plan = response["choices"][0]["message"]["content"].strip()
            lesson_plans[module] = lesson_plan
            progress_bar.progress((i + 1) / len(selected_modules))
        
        status_text.text("All lesson plans generated!")
        return lesson_plans

def main():
    st.set_page_config(
        page_title="AI Course Development",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("AI Course Development Platform")
    st.write("Enter a subject to generate a complete course curriculum with lesson plans")
    
    with st.form("course_input_form"):
        subject_input = st.text_input("Enter course subject or theme:", "Machine Learning")
        module_count = st.number_input("Enter number of modules:", min_value=1, max_value=20, value=3)
        submit_button = st.form_submit_button("Generate Course")
    
    if submit_button:
        flow = CourseDevelopmentFlow()
        flow.subject_input = subject_input
        flow.module_count = module_count
        
        with st.spinner("Creating your course..."):
            st.subheader("Course Development Process")
            lesson_plans = flow.kickoff()
        
        st.subheader("Detailed Lesson Plans")
        for i, (module, plan) in enumerate(lesson_plans.items()):
            with st.expander(f"Module {i+1}: {module}"):
                st.markdown(plan)
        
        st.subheader("Export Course Materials")
        course_data = {
            "subject": subject_input,
            "modules": [{"title": module, "lesson_plan": plan} for module, plan in lesson_plans.items()]
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Download JSON",
                data=json.dumps(course_data, indent=2),
                file_name=f"{subject_input.replace(' ', '_')}_course.json",
                mime="application/json"
            )
        with col2:
            markdown_content = f"# {subject_input} Course Curriculum\n\n"
            for i, (module, plan) in enumerate(lesson_plans.items()):
                markdown_content += f"## Module {i+1}: {module}\n\n{plan}\n\n"
            st.download_button(
                label="Download Markdown",
                data=markdown_content,
                file_name=f"{subject_input.replace(' ', '_')}_course.md",
                mime="text/markdown"
            )
        with col3:
            # Generate the PDF once and provide a persistent download button.
            pdf_bytes = generate_pdf_content(subject_input, lesson_plans)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{subject_input.replace(' ', '_')}_course.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
