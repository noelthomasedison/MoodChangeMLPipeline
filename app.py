from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from components.theme import load_css, load_js
from src.config import MOOD_COLS, RAW_DATA_PATH
from src.pipeline import refresh_payload
from src.repository import add_observation, initialize_database

load_dotenv()
st.set_page_config(page_title='Prorizon Athlete App', page_icon='🧠', layout='wide')
load_css()
load_js()
initialize_database(RAW_DATA_PATH)

payload0 = refresh_payload()
athlete_ids = payload0.get('athletes', [])
athlete_options = payload0.get('athlete_options', {})
if 'selected_athlete' not in st.session_state and athlete_ids:
    st.session_state.selected_athlete = athlete_ids[0]

payload = refresh_payload(st.session_state.get('selected_athlete'))
athlete_ids = payload.get('athletes', athlete_ids)
athlete_options = payload.get('athlete_options', athlete_options)
selected = st.sidebar.selectbox('Athlete profile', athlete_ids, index=athlete_ids.index(payload['athlete_id']) if athlete_ids else 0, format_func=lambda x: athlete_options.get(x, x))
st.session_state.selected_athlete = selected
payload = refresh_payload(selected)
page = st.sidebar.radio('View', ['Athlete App', 'Coach Overview', 'New Check-in', 'About'])

st.markdown("""
<div class='app-shell'>
  <div class='hero-app'>
    <div>
      <div class='eyebrow'>Athlete wellbeing companion</div>
      <h1>Prorizon</h1>
      <p>Helping athletes and coaches understand how training, recovery and routine shape how someone feels.</p>
    </div>
    <div class='hero-right'>
      <div class='hero-badge'>Mood-aware</div>
      <div class='hero-badge'>Real-time check-ins</div>
      <div class='hero-badge'>Coach-ready insights</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if page == 'Athlete App':
    st.markdown(f"<div class='section-title'>Today for {payload['athlete_name']}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown(f"<div class='mood-card'><div class='mood-label'>Current mood</div><div class='mood-value'>{payload['current_mood'] or 'Unknown'}</div><div class='mood-zone'>{payload['current_zone']}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='mini-card'><div class='mini-label'>Previous check-in</div><div class='mini-value'>{payload['previous_mood'] or 'No mood logged'}</div><div class='mini-sub'>{payload['previous_zone']}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='mini-card'><div class='mini-label'>Check-in context</div><div class='mini-value'>{payload['current_context']}</div><div class='mini-sub'>latest entry</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary-card'><h3>Today’s insight</h3><p>{payload['summary']}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>What may be influencing this today</div>", unsafe_allow_html=True)
    cards = st.columns(3)
    for col, theme in zip(cards, payload['driver_themes'][:3] or [{'theme': 'No clear driver', 'headline': 'No strong signal yet', 'detail': 'The latest check-in does not show one dominant driver.', 'score': 0}]):
        col.markdown(f"<div class='driver-card'><div class='driver-theme'>{theme['theme']}</div><div class='driver-headline'>{theme['headline']}</div><div class='driver-detail'>{theme['detail']}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Your recent pattern</div>", unsafe_allow_html=True)
    hist = payload['history']
    trend_cols = st.columns(2)
    if payload['trend_quality']['sleep']['show']:
        fig = px.line(hist, x='date', y='sleep_hours', markers=True, title='Sleep pattern')
        fig.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))
        trend_cols[0].plotly_chart(fig, use_container_width=True)
    else:
        trend_cols[0].info(payload['trend_quality']['sleep']['reason'])
    if payload['trend_quality']['hrv']['show']:
        fig2 = px.line(hist, x='date', y='HRV', markers=True, title='Body signal pattern (HRV)')
        fig2.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10))
        trend_cols[1].plotly_chart(fig2, use_container_width=True)
    else:
        trend_cols[1].info(payload['trend_quality']['hrv']['reason'])

    mood_timeline = hist[['date', 'predicted_primary_mood', 'predicted_zone', 'transition_flag']].tail(10).copy()
    mood_timeline.columns = ['Date', 'Mood', 'Emotional zone', 'Change detected']
    st.dataframe(mood_timeline, use_container_width=True, hide_index=True)

elif page == 'Coach Overview':
    st.markdown("<div class='section-title'>Who needs attention today</div>", unsafe_allow_html=True)
    alerts = payload['alerts']
    if alerts:
        cols = st.columns(min(3, len(alerts)))
        for col, alert in zip(cols, alerts[:3]):
            col.markdown(f"<div class='alert-card'><div class='alert-name'>{alert['athlete_name']}</div><div class='alert-mood'>{alert['current_mood']}</div><div class='alert-zone'>{alert['current_zone']}</div><div class='alert-copy'>Previously: {alert['previous_mood']}</div><div class='alert-copy'>{alert['change']}</div></div>", unsafe_allow_html=True)
    else:
        st.success('No athletes are currently flagged for extra attention.')

    st.markdown("<div class='section-title'>Team snapshot</div>", unsafe_allow_html=True)
    snap_cols = st.columns(4)
    for col, zone in zip(snap_cols, ['Energised & Driven', 'Calm & Balanced', 'Stressed & Overloaded', 'Drained & Low Energy']):
        count = payload['zone_snapshot'].get(zone, 0)
        col.markdown(f"<div class='mini-card'><div class='mini-label'>{zone}</div><div class='mini-value'>{count}</div><div class='mini-sub'>athletes</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Team overview</div>", unsafe_allow_html=True)
    coach_df = payload['coach_df'][['athlete_name', 'current_mood', 'current_zone', 'previous_mood', 'change', 'updated']].copy()
    coach_df.columns = ['Athlete', 'Current mood', 'Emotional zone', 'Previous mood', 'What changed', 'Updated']
    st.dataframe(coach_df, use_container_width=True, hide_index=True)

elif page == 'New Check-in':
    st.markdown("<div class='section-title'>Add a new check-in</div>", unsafe_allow_html=True)
    with st.form('checkin'):
        athlete_id = st.selectbox('Athlete', athlete_ids, format_func=lambda x: athlete_options.get(x, x))
        obs_time = st.datetime_input('Check-in time', value=datetime.now())
        selected_moods = st.multiselect('How are they feeling?', MOOD_COLS)
        checkin_type = st.selectbox('Context', ['pre_training', 'post_training', 'end_of_day', 'regular'])
        col1, col2, col3 = st.columns(3)
        performance = col1.slider('Performance', 0, 100, 55)
        tiredness = col2.slider('Tiredness', 0, 100, 45)
        concentrate = col3.slider('Focus', 0, 100, 50)
        hydration = col1.slider('Hydration', 0, 100, 55)
        nutrition = col2.slider('Nutrition', 0, 100, 55)
        steps = col3.number_input('Steps', min_value=0, value=7000)
        sleep_hours = col1.slider('Sleep hours', 0.0, 12.0, 7.0, 0.5)
        hrv = col2.number_input('HRV', min_value=0.0, value=55.0)
        rest_hr = col3.number_input('Resting heart rate', min_value=0.0, value=60.0)
        submitted = st.form_submit_button('Save check-in and refresh')

    if submitted:
        row = {
            'User ID': athlete_id,
            'Gender': 'unknown',
            'DOB': None,
            'Weight': 70.0,
            'Height': 170.0,
            'Date/Time': pd.Timestamp(obs_time, tz=timezone.utc).isoformat(),
            'Type': checkin_type,
            'Performance': performance,
            'Tiredness': tiredness,
            'Concentrate': concentrate,
            'Social': 50,
            'SorenessUp': 25,
            'SorenessLo': 25,
            'Hydration': hydration,
            'Nutrition': nutrition,
            'Steps': steps,
            'Sleep': sleep_hours * 3600,
            'HRV': hrv,
            'Sleep Heart Rate': 55.0,
            'Resting Heart Rate': rest_hr,
        }
        for mood in MOOD_COLS:
            row[mood] = mood in selected_moods
        add_observation(row)
        st.session_state.selected_athlete = athlete_id
        updated = refresh_payload(athlete_id)
        st.success('Check-in saved. The athlete view is now updated.')
        st.markdown(f"<div class='summary-card'><h3>Updated insight</h3><p>{updated['summary']}</p></div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='section-title'>About this experience</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='summary-card'>
      <h3>What this app does</h3>
      <p>This app helps athletes and coaches connect everyday wellbeing signals with how an athlete feels. It uses recent check-ins, recovery signals, training context and body-based measures to understand the current mood, identify meaningful changes and explain what may be influencing that shift.</p>
      <ul>
        <li>Focuses on mood rather than abstract scores</li>
        <li>Updates immediately after a new check-in</li>
        <li>Shows simple, human-friendly emotional zones</li>
        <li>Highlights only the most relevant drivers</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
