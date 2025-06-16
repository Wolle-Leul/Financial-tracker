from datetime import datetime, timedelta
import calendar
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import holidays
import PyPDF2 as PDF 
import pandas as pd 
import re
import numpy as np
from transformers import pipeline
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # remove password after verification
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("😕 Incorrect password")
        return False
    else:
        return True

if check_password():
    st.success("Welcome! You are logged in.")
    # Place rest of your app code here
    # I started with knowing whats the day today
    today = datetime.today()
    today_month = today.month
    today_year = today.year
    today_date = today.day
    today_week = today.isocalendar()[1]
    # Iterate through holidays in Poland for the year 2024
    holiday_data = []
    for date, reason in sorted(holidays.Poland(years=today_year).items()):
        holiday_data.append({'Date': date, 'Reason': reason})
    # Convert the list of dictionaries into a DataFrame
    holiday_df = pd.DataFrame(holiday_data)
    Inavitables = pd.DataFrame({'Catagory': ["Utlities", "Utlities", "Entertainment", "Housing", "Groceries"],
                                'Sub-Catagory': ["Phone", "Utlities", "You Tube", "Rent", "Groceries"],
                                'Ammount': [350.00, 700.00, 0.00, 20000.00, 8000.00],
                                'Deadline': [10, 12, 16, 12, 28]})
    Incomes = pd.DataFrame({'Source': ["Infy"],
                            'FY': ["FY2324"],
                            'Ammount': [50000.00], })
    st.set_page_config(page_title="Tracker", page_icon=":chart_with_upwards_trend:", layout="wide")
    # Sidebar for month and year selection
    month = st.sidebar.selectbox("Select Month", range(1, 13),
                                 format_func=lambda x: datetime(1900, x, 1).strftime('%B'), index=today_month - 1)
    year = st.sidebar.selectbox("Select Year", range(today_year - 1, today_year + 1), index=1)
    # slicers placed at the side bar
    # for catagory
    st.sidebar.header("Catagory")
    Catagory = st.sidebar.multiselect("Pick your Catagory", Inavitables["Catagory"].unique())
    if not Catagory:
        Inavitables2 = Inavitables.copy()
    else:
        Inavitables2 = Inavitables[Inavitables["Catagory"].isin(Catagory)]
    # for Sub-catagory
    st.sidebar.header("Sub-Catagory")
    SCatagory = st.sidebar.multiselect("Pick your Sub-Catagory", Inavitables2["Sub-Catagory"].unique())
    if not SCatagory:
        Inavitables3 = Inavitables2.copy()
    else:
        Inavitables3 = Inavitables2[Inavitables2["Sub-Catagory"].isin(SCatagory)]
    holiday = holiday_df['Date'].tolist()
    holiday = [date.strftime('%Y-%m-%d') for date in holiday]


    # Function to generate Salary day
    def generate_salaryday(year, month, holidays):
        # Mark salary date
        salary_date = 10  # Default salary date is 10th of every month
        salary_date_dt = datetime(year, month, salary_date)
        if salary_date_dt.weekday() < 5:  # If salary date is on a weekday
            if salary_date_dt.strftime('%Y-%m-%d') in holidays:  # If salary date is a holiday
                if salary_date_dt.weekday() == 0:  # If salary date is on Monday
                    salary_date_dt -= timedelta(days=3)  # Mark (salary date - 3) as green
                else:
                    salary_date_dt -= timedelta(days=1)  # Mark (salary date - 1) as green
            else:
                if salary_date_dt.weekday() == 0:  # If salary date is on Monday
                    salary_date_dt -= timedelta(days=1)  # Mark (salary date - 1) as green
        else:  # If salary date is on a weekend
            if salary_date_dt.weekday() == 5:  # If salary date is on Saturday
                salary_date_dt -= timedelta(days=1)  # Mark (salary date - 1) as green
            else:
                salary_date_dt -= timedelta(days=2)  # Mark (salary date - 2) as green
        return salary_date_dt


    # Function to generate calendar plot with holiday markings
    def generate_calendar(year, month, holidays, salary, today):
        # Determine the weekday of the first day of the month
        first_day_of_month = datetime(year, month, 1)
        start_day = first_day_of_month.weekday()  # 0: Monday, 6: Sunday
        # Determine the number of days in the month
        num_days = calendar.monthrange(year, month)[1]
        # Calculate the number of weeks needed to represent all days of the month
        num_weeks = (start_day + num_days) // 7 + 1 if (start_day + num_days) % 7 != 0 else (start_day + num_days) // 7
        fig = go.Figure()
        # Add day numbers
        fig.add_trace(go.Scatter(x=[(i + start_day) % 7 for i in range(num_days)],
                                 y=[num_weeks - 1 - (i + start_day) // 7 for i in range(num_days)],
                                 mode='text',
                                 text=[str(i + 1) for i in range(num_days)],
                                 textfont=dict(color='white', size=16),
                                 hoverinfo='skip'))
        # Mark holidays
        for holiday in holidays:
            holiday_date = datetime.strptime(holiday, "%Y-%m-%d")
            if holiday_date.year == year and holiday_date.month == month:
                day = (holiday_date.day + start_day - 1) % 7
                week = num_weeks - 1 - (holiday_date.day + start_day - 1) // 7
                fig.add_trace(go.Scatter(x=[day], y=[week],
                                         mode='markers',
                                         marker=dict(size=15, color='red'),
                                         hoverinfo='skip'))
        salary_date_dt = generate_salaryday(year, month, holidays)
        if salary_date_dt.month == month:  # Check if the adjusted salary date is in the current month
            day = (salary_date_dt.day + start_day - 1) % 7
            week = num_weeks - 1 - (salary_date_dt.day + start_day - 1) // 7
            fig.add_trace(go.Scatter(x=[day], y=[week],
                                     mode='markers',
                                     marker=dict(size=15, color='green'),
                                     hoverinfo='skip'))
        # mark today
        if today.month == month and today_date:  # Check if the adjusted salary date is in the current month
            day = (today.day + start_day - 1) % 7
            week = num_weeks - 1 - (today.day + start_day - 1) // 7
            fig.add_trace(go.Scatter(x=[day], y=[week],
                                     mode='markers',
                                     marker=dict(size=15, color='blue'),
                                     hoverinfo='skip'))
        # Update layout
        fig.update_layout(title=f'({datetime(year, month, 1).strftime("%B %Y")})',
                          showlegend=False,
                          xaxis=dict(tickmode='array', tickvals=list(range(7)),
                                     ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                          yaxis=dict(tickmode='array', tickvals=list(range(num_weeks)),
                                     ticktext=['Week {}'.format(i + 1) for i in
                                               sorted(range(num_weeks), reverse=True)]))
        return fig


    # now I would like to capture current and previous month salary dates
    salary_this_month = generate_salaryday(today_year, today_month, holiday)
    if salary_this_month.month == today_month:
        if salary_this_month.day <= today_date:
            salary_prev_month = salary_this_month
        elif today_month == 1:
            salary_prev_month = generate_salaryday(today_year - 1, 12, holiday)
        else:
            salary_prev_month = generate_salaryday(today_year, today_month - 1, holiday)
    else:
        salary_prev_month = generate_salaryday(today_year, today_month - 1, holiday)

    # then due and past salary date is calculated as
    if salary_this_month.month == today_month:
        if salary_this_month.day <= today_date:
            if today_month == 12:
                due_salary_date = generate_salaryday(today_year + 1, 1, holiday)
            else:
                due_salary_date = generate_salaryday(today_year, today_month + 1, holiday)
        else:
            due_salary_date = salary_this_month
    else:
        due_salary_date = generate_salaryday(today_year, today_month + 1, holiday)

    # I would like to  calculate measures regarding the salary dates now
    days_till_NextSalary = (due_salary_date - today).days
    days_from_PrevSalary = (today - salary_prev_month).days
    # I split the screen into 3 columns
    total_width = 1000
    # calculate how much should be left for groceries
    Inav_Gr = Inavitables[Inavitables['Sub-Catagory'].str.contains("Groceries")]
    gr_total = Inav_Gr['Ammount'].astype(float).sum()
    groceries = (gr_total / (days_till_NextSalary + days_from_PrevSalary)) * days_till_NextSalary
    # net-of-net income
    net_of_net = (Incomes['Ammount'].astype(float).sum()) - (Inavitables['Ammount'].astype(float).sum())
    # future
    Target_incom = (Inavitables['Ammount'].astype(int).sum()) / 0.45
    left_to_target = (Incomes['Ammount'].astype(float).sum()) - Target_incom
    Target_NetofNet = net_of_net / 0.45
    needs = ((Inavitables['Ammount'].astype(int).sum()) / (Incomes['Ammount'].astype(float).sum())) * 100
    TargetVsActual = ((Target_NetofNet - net_of_net) / Target_NetofNet) * 100
    ddtdate = due_salary_date.date


    # now I will create a pivot table
    def format_amount(value):
        return '{:,.2f}'.format(value)


    Inavitables3.loc[Inavitables3['Sub-Catagory'].str.contains("Groceries"), 'Ammount'] = groceries
    Inavitables3['Deadline'] = pd.to_datetime(Inavitables3['Deadline'], format='%d')
    Inavitables3['Deadline'] = Inavitables3['Deadline'].apply(
        lambda x: x.replace(month=salary_prev_month.month, year=salary_prev_month.year))
    Inavitables3.loc[Inavitables3['Sub-Catagory'].str.contains("Groceries"), 'Deadline'] = pd.to_datetime(
        due_salary_date)
    Inavitables3['Deadline'] = Inavitables3['Deadline'].dt.date
    Inavitables3 = Inavitables3.drop('Catagory', axis=1)
    # Calculate the relative widths of the columns
    c1w = total_width // 3
    c2w = total_width // 3
    c3w = total_width // 3
    col1, col2, col3 = st.columns((c1w, c2w, c3w))

    with col1:
        # Generate and display the calendar plot with holiday markings
        st.text(str(days_till_NextSalary) + " days left for infy")
        with st.expander("Calendar", expanded=False):
            st.subheader("📅 Calendar")
            st.plotly_chart(generate_calendar(year, month, holiday, holiday, today), use_container_width=True)
    sub_col2_1, sub_col2_2 = col2.columns(2)
    with sub_col2_1:
        new_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Due date </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.text(str(due_salary_date.strftime("%Y-%m-%d")))
        new_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Groceries </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.text(str(f"{round(groceries, 2):,}") + " $ ")
        new_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 15px;">Net of Net </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.text(str(f"{round(net_of_net, 2):,}") + " $")
    with sub_col2_2:
        new_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">Need%</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.text(str(f"{round(needs, 2):,}") + " %")
        new_title = '<p style="font-family:sans-serif; color:#89CFF0; font-size: 16px;">ΔTarget%</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.text(str(f"{round(TargetVsActual, 2):,}") + " %")
    with col2:
        # Style the DataFrame with a gradient background
        styled_df = Inavitables3.style.background_gradient(subset=['Ammount'], cmap='Blues')
        # Display the styled DataFrame as a table
        styled_df = styled_df.format({'Ammount': format_amount})
        with st.expander("To Pay", expanded=False):
            styled_df_html = styled_df.to_html(index=False, escape=False)
            st.write(styled_df, unsafe_allow_html=True)
    Inc_header = Incomes['Source'].tolist()
    Inc_ammount = Incomes['Ammount'].tolist()
    Inv_header = Inavitables['Sub-Catagory'].tolist()
    Inv_ammount = Inavitables['Ammount'].tolist()
    I = 0
    for l in Inc_ammount:
        I += l
    J = 0
    for l in Inv_ammount:
        J += l
    K = I - J
    Inv_ammount.append(K)
    source = Inc_header + ["Total"] + Inv_header + ["Left"]
    Inc_ammount.append(0)
    value = Inc_ammount + Inv_ammount

    source_indexes = [i for i in range(len(source))]
    Target_indexes = [i for i in range(len(source))]
    center = source.index("Total")  # Find the index of "Total" in the list
    for i in range(len(source_indexes)):
        if i <= center:
            source_indexes[i] = i
        else:
            source_indexes[i] = center
    for i in range(len(source_indexes)):
        if i < center:
            Target_indexes[i] = center
        else:
            Target_indexes[i] = i
    link = dict(source=source_indexes, target=Target_indexes, value=value)
    node = dict(label=source, pad=20, thickness=30, color="#89CFF0")
    data = go.Sankey(link=link, node=node, orientation="v")
    with col3:
        fig = go.Figure(data)
        fig.update_layout(margin=dict(l=0, r=0, t=5, b=5), width=600)  # Adjust width as needed
        st.plotly_chart(fig, use_container_width=True, unsafe_allow_html=True)

