"""
Streamlit UI for FocusGuard
--------------------------
A dashboard for monitoring focus, viewing statistics, and configuring the application.
"""

import streamlit as st
import pandas as pd
import altair as alt
import time
import uuid
from datetime import datetime, timedelta
from focus_guard import FocusGuard, DEFAULT_CONFIG
import json


# MUST be the first Streamlit command
st.set_page_config(
    page_title="FocusGuard",
    page_icon="🌿",
    layout="wide"
)

def create_streamlit_app():
    # Initialize session state
    if 'focus_guard' not in st.session_state:
        st.session_state.focus_guard = FocusGuard()
        st.session_state.monitoring = False
        st.session_state.last_update = time.time()
        st.session_state.config = DEFAULT_CONFIG.copy()
        st.session_state.show_settings = False
    
    # Initialize button keys separately - only if they don't exist
    if 'settings_button_key' not in st.session_state:
        st.session_state.settings_button_key = f"settings_button_{uuid.uuid4().hex}"
    if 'save_button_key' not in st.session_state:
        st.session_state.save_button_key = "save_settings_button"
    if 'start_button_key' not in st.session_state:
        st.session_state.start_button_key = "start_monitoring_button"
    if 'stop_button_key' not in st.session_state:
        st.session_state.stop_button_key = "stop_monitoring_button"
    
    # Header
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🌿 FocusGuard")
        st.subheader("AI-powered Distraction Blocker")
    with col2:
        if st.button("⚙️ Settings", use_container_width=True, key=st.session_state.settings_button_key):
            st.session_state.show_settings = not st.session_state.show_settings
    
    # Settings panel
    if st.session_state.show_settings:
        with st.expander("Settings", expanded=True):
            st.subheader("Monitoring Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                interval = st.slider("Screenshot Interval (seconds)", 
                                     min_value=5, max_value=30, 
                                     value=st.session_state.config["screenshot_interval"])
                
                threshold = st.slider("Distraction Threshold", 
                                      min_value=1, max_value=10, 
                                      value=st.session_state.config["distraction_threshold"],
                                      help="Number of consecutive distractions before intervention")
            
            with col2:
                notification_freq = st.slider("Notification Frequency (minutes)", 
                                             min_value=1, max_value=15, 
                                             value=st.session_state.config["notification_frequency"])
                
                max_text = st.slider("Max Text Length", 
                                    min_value=100, max_value=1000, 
                                    value=st.session_state.config["max_text_length"])
            
            st.subheader("Intervention Settings")
            col1, col2 = st.columns(2)
            with col1:
                enable_voice = st.checkbox("Enable Voice Alerts", 
                                          value=st.session_state.config["enable_voice_alerts"])
            
            with col2:
                enable_blocking = st.checkbox("Enable App Blocking", 
                                             value=st.session_state.config["enable_app_blocking"])
            
            st.subheader("Privacy Settings")
            col1, col2 = st.columns(2)
            with col1:
                store_text = st.checkbox("Store Raw Text", 
                                        value=st.session_state.config["privacy"]["store_raw_text"])
            
            with col2:
                anonymize = st.checkbox("Anonymize Personal Data", 
                                       value=st.session_state.config["privacy"]["anonymize_personal_data"])
            
            if st.button("Save Settings", key=st.session_state.save_button_key):
                # Update config
                new_config = st.session_state.config.copy()
                new_config["screenshot_interval"] = interval
                new_config["distraction_threshold"] = threshold
                new_config["notification_frequency"] = notification_freq
                new_config["max_text_length"] = max_text
                new_config["enable_voice_alerts"] = enable_voice
                new_config["enable_app_blocking"] = enable_blocking
                new_config["privacy"]["store_raw_text"] = store_text
                new_config["privacy"]["anonymize_personal_data"] = anonymize
                
                # Update session state and focus guard config
                st.session_state.config = new_config
                st.session_state.focus_guard.update_config(new_config)
                st.success("Settings saved successfully!")
    
    # Main dashboard
    st.markdown("---")
    
    # Status and control
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if not st.session_state.monitoring:
            if st.button("▶️ Start Monitoring", use_container_width=True, key=st.session_state.start_button_key):
                st.session_state.focus_guard.start_monitoring()
                st.session_state.monitoring = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Monitoring", use_container_width=True, key=st.session_state.stop_button_key):
                st.session_state.focus_guard.stop_monitoring()
                st.session_state.monitoring = False
                st.rerun()
    
    with col2:
        status = "✅ Active" if st.session_state.monitoring else "⏸️ Paused"
        status_color = "green" if st.session_state.monitoring else "gray"
        st.markdown(f"<h3 style='color:{status_color};'>Status: {status}</h3>", unsafe_allow_html=True)
    
    # Stats cards
    st.markdown("---")
    st.subheader("Focus Statistics")
    
    # Auto-update stats
    if st.session_state.monitoring and time.time() - st.session_state.last_update > 5:
        st.session_state.last_update = time.time()
        st.rerun()
    
    # Get current stats
    stats = st.session_state.focus_guard.get_stats()
    recent_activities = st.session_state.focus_guard.get_recent_activities(20)

     # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Productivity Score", 
            f"{stats['productivity_score']}%",
            delta=None
        )
    
    with col2:
        tracked_time = stats['total_tracked_time']
        time_display = f"{tracked_time:.1f} min" if tracked_time < 60 else f"{tracked_time/60:.1f} hours"
        st.metric(
            "Time Tracked", 
            time_display,
            delta=None
        )
    
    with col3:
        st.metric(
            "Interventions", 
            stats['intervention_count'],
            delta=None
        )
    
    with col4:
        # Calculate distraction percentage
        total_activities = sum(stats['activity_breakdown'].values()) if stats['activity_breakdown'] else 1
        distraction_count = sum(stats['activity_breakdown'].get(k, 0) for k in ['SOCIAL_MEDIA', 'ENTERTAINMENT', 'GAMES'])
        distraction_pct = int((distraction_count / total_activities) * 100) if total_activities else 0
        
        st.metric(
            "Distraction %", 
            f"{distraction_pct}%",
            delta=None
        )

    
    # Activity breakdown
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Activity Breakdown")
        
        # Prepare data for chart
        breakdown = stats['activity_breakdown']
        if breakdown:
            chart_data = pd.DataFrame({
                'Activity': list(breakdown.keys()),
                'Minutes': [value * (st.session_state.config["screenshot_interval"] / 60) for value in breakdown.values()]
            })
            
            # Create bar chart
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Activity:N', sort='-y'),
                y='Minutes:Q',
                color=alt.Color('Activity:N', scale=alt.Scale(
                    domain=['STUDY', 'WORK', 'NEUTRAL', 'SOCIAL_MEDIA', 'ENTERTAINMENT', 'GAMES'],
                    range=['#2ca02c', '#1f77b4', '#7f7f7f', '#d62728', '#ff7f0e', '#9467bd']
                ))).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No activity data available yet. Start monitoring to collect data.")
    
    with col2:
        st.subheader("Productivity Trend")
        
        activities = st.session_state.focus_guard.get_recent_activities(100)
        
        if activities:
            # Group by timestamp (rounded to nearest 5 minutes)
            trend_data = []
            for activity in activities:
                timestamp = datetime.strptime(activity['timestamp'], "%Y-%m-%d %H:%M:%S")
                rounded_time = timestamp.replace(minute=5*(timestamp.minute//5), second=0)
                
                # Classify as productive or not
                is_productive = 1 if activity['activity_type'] in ['STUDY', 'WORK', 'NEUTRAL'] else 0
                
                trend_data.append({
                    'time': rounded_time,
                    'productive': is_productive
                })
            
            # Aggregate by time period
            df = pd.DataFrame(trend_data)
            if not df.empty:
                df_grouped = df.groupby('time').agg(
                    productivity=('productive', 'mean'),
                    count=('productive', 'count')
                ).reset_index()
                
                df_grouped['productivity'] = df_grouped['productivity'] * 100
                
                # Create line chart
                line_chart = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x='time:T',
                    y=alt.Y('productivity:Q', scale=alt.Scale(domain=[0, 100]), title='Productivity %'),
                    tooltip=['time:T', 'productivity:Q', 'count:Q']
                ).properties(height=250)
                
                st.altair_chart(line_chart, use_container_width=True)
            else:
                st.info("Collecting trend data...")
        else:
            st.info("No activity data available yet. Start monitoring to collect data.")
    
    # Recent activities log
    st.markdown("---")
    st.subheader("Recent Activities")
    
    activities = st.session_state.focus_guard.get_recent_activities(20)
    
    if activities:
        # Create a DataFrame for the activities
        activities_df = pd.DataFrame(activities)
        
        # Format the activities for display
        for _, activity in activities_df.iterrows():
            activity_type = activity['activity_type']
            timestamp = activity['timestamp']
            window_title = activity.get('window_title', '')
            content = activity.get('content_preview', '')
            action = activity.get('action_taken', '')
            
            # Color coding
            if activity_type in ['STUDY', 'WORK']:
                color = 'green'
            elif activity_type in ['NEUTRAL']:
                color = 'blue'
            elif activity_type == 'GAMES':
                color = 'purple'
            else:
                color = 'red'
            
            # Action icon
            action_icon = ''
            if action == 'notification':
                action_icon = '🔔'
            elif action == 'voice_alert':
                action_icon = '🔊'
            elif action == 'tab_closed':
                action_icon = '🚫'
            
            # Display activity
            st.markdown(f"""
            <div style="margin-bottom: 15px; padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style='color:{color}; font-weight:bold;'>{activity_type}</span>
                    <span style='color:gray; font-size:0.8em;'>{timestamp}</span>
                </div>
                <div style="margin-top: 5px;">
                    <b>{window_title[:50]}{'...' if len(window_title) > 50 else ''}</b> {action_icon}
                </div>
                <div style="font-size:0.8em; color:#666; margin-top:2px;">
                    {content[:70]}{'...' if len(content) > 70 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No activity data available yet. Start monitoring to collect data.")

if __name__ == "__main__":
    create_streamlit_app()