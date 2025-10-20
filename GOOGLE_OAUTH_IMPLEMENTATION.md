# Google OAuth Implementation Summary

## Changes Made

### 1. Updated requirements.txt
- Changed `streamlit` to `streamlit[auth]` to include Authlib dependency

### 2. Created Configuration Files
- Created `.streamlit/secrets.toml` with Google OAuth configuration template
- Created `.streamlit/secrets.toml.example` as a reference file
- Created `.env.example` with application password configuration template

### 3. Modified Authentication Logic in RAG.py
- Enhanced Google OAuth authentication with domain and account restrictions
- Added second-layer password authentication using HMAC comparison
- Implemented `st.login()` for authentication flow
- Used `st.user` to access user information after authentication
- Added `st.logout()` functionality with proper session cleanup
- Added `is_authorized_account()` function for account-level restrictions
- Added `check_password()` function for second-layer authentication

### 4. Updated UI
- Added user information display (name and email) in the application header
- Added logout button with proper session cleanup
- Improved login screen with clear instructions
- Added password authentication screen after Google OAuth

### 5. Documentation
- Updated README.md with comprehensive two-layer authentication setup instructions
- Added step-by-step guide for creating Google OAuth credentials
- Documented domain and account restriction configuration
- Documented password authentication configuration

## How It Works

1. **Two-Layer Authentication Flow**: 
   - Users click "Sign in with Google" button
   - `st.login()` initiates the OAuth flow
   - Users are redirected to Google for authentication
   - After authentication, users are redirected back to the app
   - If domain/account restrictions are configured, only authorized users can proceed
   - Users must then enter application password for second-layer authentication

2. **Domain Restrictions**:
   - Configured via `authorized_domains` in secrets.toml or `AUTHORIZED_DOMAINS` environment variable
   - Only users from specified domains can access the application
   - Unauthorized users are shown an error message and logout button

3. **Account Restrictions**:
   - Configured via `AUTHORIZED_ACCOUNTS` environment variable
   - Only specific email addresses can access the application
   - Unauthorized users are shown an error message and logout button

4. **Password Authentication**:
   - Configured via `AUTH_EMAIL` and `AUTH_PASSWORD` environment variables
   - Uses HMAC comparison for secure password checking
   - Session state management to track authentication status

5. **User Information**:
   - Displays user's name and email in the application header
   - Provides a logout button for ending the session
   - Clears password authentication state on logout

## Configuration

To configure two-layer authentication:

1. Create Google OAuth credentials in Google Cloud Console
2. Rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
3. Fill in your Google OAuth credentials
4. Optionally configure authorized domains in secrets.toml
5. Set `AUTHORIZED_ACCOUNTS` environment variable for account-level restrictions
6. Set `AUTH_EMAIL` and `AUTH_PASSWORD` environment variables for password authentication
7. Run the application with `streamlit run RAG.py`

## Security Features

- Automatic cookie management with secure storage
- Session handling with automatic expiration
- Domain-based access control
- Account-level access control
- Secure OAuth flow with state and nonce validation
- Cookie encryption with user-provided secret
- HMAC-based password authentication
- Proper session cleanup on logout