
"""
Internal state object stores persistent per-session variables.

This fixes a situation where a button within a button may 
reset the whole non-cached functions as seen here:
https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/3

Solution taken from: 
https://gist.github.com/tvst/faf057abbedaccaa70b48216a1866cdd

Forum thread on the topic shows an updated st_state_patch.py and a promise of an official patch:
https://discuss.streamlit.io/t/programmable-state-for-streamlit/2052




Hack to add per-session state to Streamlit.
Usage
-----
>>> import SessionState
>>>
>>> session_state = SessionState.get(user_name='', favorite_color='black')
>>> session_state.user_name
''
>>> session_state.user_name = 'Mary'
>>> session_state.favorite_color
'black'
Since you set user_name above, next time your script runs this will be the
result:
>>> session_state = get(user_name='', favorite_color='black')
>>> session_state.user_name
'Mary'




A good example to show how it works:
https://discuss.streamlit.io/t/update-slider-value/372/2
---------------------------------------------------------
import streamlit as st

# SessionState module from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState

state = SessionState.get(position=0)

# The slider needs to come after the button, to make sure the first increment
# works correctly. So we create a placeholder for it here first, and fill it in
# later.
widget = st.empty()

if st.button('Increment position'):
    state.position += 1

state.position = widget.slider('Position', 0, 100, state.position)





Another example script from: 
https://gist.github.com/tvst/faf057abbedaccaa70b48216a1866cdd
----------------------

import streamlit as st
# Normally you'd import the file above here.
# import SessionState

st.sidebar.title("Pages")
radio = st.sidebar.radio(label="", options=["Set A", "Set B", "Add them"])

# Normally you'd do this:
#session_state = SessionState.get(a=0, b=0)
# ...but since we're not importing SessionState, we'll just do:
session_state = get(a=0, b=0)  # Pick some initial values.

if radio == "Set A":
    session_state.a = float(st.text_input(label="What is a?", value=session_state.a))
    st.write(f"You set a to {session_state.a}")
elif radio == "Set B":
    session_state.b = float(st.text_input(label="What is b?", value=session_state.b))
    st.write(f"You set b to {session_state.b}")
elif radio == "Add them":
    st.write(f"a={session_state.a} and b={session_state.b}")
    button = st.button("Add a and b")
    if button:
        st.write(f"a+b={session_state.a+session_state.b}")

"""
import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.
        Parameters
        ----------
        **kwargs : any
            Default values for the session state.
        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'
        """
        for key, val in kwargs.items():
            setattr(self, key, val)


def get(**kwargs):
    """Gets a SessionState object for the current session.
    Creates a new object if necessary.
    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'
    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'
    """
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None
    
    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56        
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state