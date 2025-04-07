import streamlit as st

def notify_message(message, isValid=False):
    bg_color = "#4CAF50" if isValid else "#f44336"

    st.markdown(
        f"""
        <style>
        .flash-message {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: {bg_color};
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            z-index: 9999;
            font-weight: bold;
            animation: fadeout 4s ease-in-out forwards;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            max-width: 90vw;
            text-align: center;
            word-wrap: break-word;
        }}
        @keyframes fadeout {{
            0% {{ opacity: 1; }}
            80% {{ opacity: 1; }}
            100% {{ opacity: 0; display: none; }}
        }}
        </style>
        <div class="flash-message">{message}</div>
        """,
        unsafe_allow_html=True
    )

