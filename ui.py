def sidebar_nav_css() -> str:
    """Pin Streamlit's built-in page list to the bottom of the sidebar.

    Do not use position:fixed with a hardcoded width — that overflows the sidebar.
    """
    return """
    section[data-testid="stSidebar"] {
        position: relative !important;
        overflow-x: hidden !important;
    }
    [data-testid="stSidebarContent"] {
        position: relative !important;
        height: 100% !important;
    }
    [data-testid="stSidebarNav"] {
        position: absolute !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: auto !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        background: #10182a;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.7rem 0.85rem 1rem 0.85rem;
        z-index: 20;
    }
    [data-testid="stSidebarUserContent"] {
        padding-bottom: 5.75rem !important;
    }
    """
