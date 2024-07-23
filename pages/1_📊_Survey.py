import os
import streamlit as st
import json
from datetime import datetime
import streamlit_survey as ss

st.markdown('<h2 class="section_title">Military Spouse Experience Survey</h2>', unsafe_allow_html=True)

st.markdown('<h2 class="section_title">Military Spouse Experience Survey</h2>', unsafe_allow_html=True)
    
uploaded_file = st.file_uploader("Upload a survey response file", type="json")
if uploaded_file is not None:
    survey_data = json.load(uploaded_file)
    st.session_state['survey_data'] = survey_data
    st.success("File uploaded successfully!")

# Initialize or get the survey from session state
if 'survey' not in st.session_state:
    st.session_state['survey'] = ss.StreamlitSurvey("Military Spouse Experience Survey")
survey = st.session_state['survey']

# Create paged survey
pages = survey.pages(5, on_submit=lambda: st.success("Your responses have been recorded. Thank you!"))

with pages:
    if pages.current == 0:
        st.subheader("Personal Information")
        survey.text_input("What is your name?")
        
        st.subheader("Education")
        education = survey.radio(
            "What is your highest level of education?",
            options=["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"],
            horizontal=True
        )

        if education != "High School":
            survey.text_area(
                "Please describe your most significant educational experiences or achievements:"
            )

            survey.multiselect(
                "What challenges, if any, have you faced in your education due to being a military spouse? (Select all that apply)",
                options=[
                    "Frequent relocations",
                    "Difficulty transferring credits",
                    "Limited time due to family responsibilities",
                    "Financial constraints",
                    "Lack of consistent childcare",
                    "Other"
                ]
            )

    elif pages.current == 1:
        st.subheader("Work Experience")
        work_status = survey.multiselect(
            "What is your current employment status? (Select all that apply)",
            options=[
                "Employed full-time",
                "Employed part-time",
                "Self-employed",
                "Freelance/Contract work",
                "Unemployed, seeking work",
                "Not currently in the workforce",
                "Student",
                "Volunteer",
                "Stay-at-home parent/caregiver",
                "Retired",
                "Other"
            ]
        )

        survey.text_input("If currently employed or previously employed, what field(s) do you work in?")
        
        survey.text_area(
            "Please describe your most significant work experiences, achievements, or volunteer roles:"
        )

        survey.multiselect(
            "What challenges, if any, have you faced in your career or employment journey due to being a military spouse? (Select all that apply)",
            options=[
                "Frequent job changes due to relocations",
                "Limited job opportunities in duty station locations",
                "Difficulty advancing in career",
                "Balancing work with family responsibilities",
                "Employer bias against military spouses",
                "Licensing or certification issues across states",
                "Gaps in employment history",
                "Difficulty finding jobs that match skills/experience",
                "Limited networking opportunities",
                "Challenges with childcare",
                "Difficulty completing education or training programs",
                "Other"
            ]
        )

        survey.text_area(
            "If you've faced employment challenges, how have you adapted or what strategies have you used to overcome them?"
        )

        survey.multiselect(
            "What types of work arrangements are you most interested in? (Select all that apply)",
            options=[
                "Full-time traditional employment",
                "Part-time employment",
                "Remote work",
                "Flexible hours",
                "Self-employment/Entrepreneurship",
                "Freelance/Contract work",
                "Job sharing",
                "Seasonal work",
                "Volunteer positions",
                "Internships or apprenticeships",
                "Other"
            ]
        )

    elif pages.current == 2:
        st.subheader("Military Spouse Daily Life")
        pcs_count = survey.slider(
            "How many times have you PCSed (Permanent Change of Station) as a military spouse?",
            min_value=0,
            max_value=20,
            value=1
        )

        if pcs_count > 0:
            survey.multiselect(
                "What challenges have you faced during PCS moves? (Select all that apply)",
                options=[
                    "Finding new housing",
                    "Children's education transitions",
                    "Personal career disruptions",
                    "Making new friends/building community",
                    "Managing household goods shipments",
                    "Emotional stress",
                    "Financial strain",
                    "Other"
                ]
            )

        survey.multiselect(
            "Which of the following tasks do you regularly manage in your household? (Select all that apply)",
            options=[
                "Budgeting and finances",
                "Childcare and education",
                "Home maintenance",
                "Healthcare management",
                "Deployment preparation",
                "Community involvement",
                "Support group participation",
                "Personal career development",
                "Other"
            ]
        )

        parenting = survey.radio(
            "Are you a parent?",
            options=["Yes", "No"],
            horizontal=True
        )

        if parenting == "Yes":
            survey.multiselect(
                "What unique challenges do you face as a military parent? (Select all that apply)",
                options=[
                    "Explaining deployments to children",
                    "Managing children's emotions during separations",
                    "Finding consistent childcare",
                    "Navigating school changes during PCS",
                    "Balancing parenting with military lifestyle demands",
                    "Maintaining family traditions despite frequent moves",
                    "Other"
                ]
            )

    elif pages.current == 3:
        st.subheader("General Experience")
        survey.text_area(
            "What do you find most rewarding about being a military spouse?"
        )

        survey.text_area(
            "What is the biggest challenge you face as a military spouse?"
        )

        survey.text_area(
            "What kind of support or resources do you wish were more readily available to military spouses?"
        )

    
# Add upload/download functionality at the top
col1, col2 = st.columns(2)
with col1:
    if st.button("Complete Survey"):
        st.session_state['survey_completed'] = True
        st.session_state['survey_data'] = survey.to_json()
    
        # Save to cached_responses folder
        os.makedirs("cached_responses", exist_ok=True)
        filename = f"cached_responses/survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(st.session_state['survey_data'], f)
    
        st.success(f"Survey completed! Responses saved to {filename}")
        st.info("You can now proceed to the Persona Generator.")

with col2:
    if 'survey_completed' in st.session_state and st.session_state['survey_completed']:
        if st.download_button("Download Survey Responses", 
                                data=json.dumps(st.session_state['survey_data'], indent=2),
                                file_name="survey_responses.json",
                                mime="application/json"):
            st.success("Survey responses downloaded!")

if st.button("Complete Survey"):
    # Your survey completion logic
    st.success("Survey completed! You can now proceed to the Persona Generator.")