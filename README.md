# RAG Document Handling Application with Two-Layer Authentication

This application provides a complete solution for ingesting, processing, and querying contract documents using Retrieval-Augmented Generation (RAG) techniques.

## Two-Layer Authentication System

The application implements a two-layer authentication system:
1. **Google OAuth** - First layer for identity verification
2. **Application Password** - Second layer for additional security

### Account Restrictions

You can restrict access to the application in two ways:
1. **Domain Restrictions** - Limit access to specific email domains
2. **Account Whitelist** - Limit access to specific email addresses

## Setting up Google OAuth Authentication

To enable Google OAuth authentication, you need to configure the application with your Google OAuth credentials.

### Step 1: Create Google OAuth Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. Select "Web application" as the application type
6. Add the following authorized redirect URI:
   ```
   http://localhost:8501/oauth2callback
   ```
7. Click "Create" and note down your Client ID and Client Secret

### Step 2: Configure the Application

1. Rename the `.streamlit/secrets.toml.example` file to `.streamlit/secrets.toml`
2. Fill in your Google OAuth credentials:
   ```toml
   [auth]
   redirect_uri = "http://localhost:8501/oauth2callback"
   cookie_secret = "YOUR_RANDOM_COOKIE_SECRET_HERE"  # Generate a random secret
   
   client_id = "YOUR_GOOGLE_CLIENT_ID_HERE"
   client_secret = "YOUR_GOOGLE_CLIENT_SECRET_HERE"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   
   # Optional: Specify authorized domains (comma-separated)
   # authorized_domains = "yourcompany.com,partnercompany.com"
   ```
3. Generate a random `cookie_secret` (at least 32 characters)

### Step 3: Configure Account Restrictions

#### Domain Restrictions
To restrict access to specific Google domains:
1. Add the `authorized_domains` setting to your `secrets.toml` file
2. Specify a comma-separated list of domains that are allowed to access the application

Example:
```toml
authorized_domains = "yourcompany.com,partnercompany.com"
```

#### Account Whitelist
To restrict access to specific Google accounts:
1. Set the `AUTHORIZED_ACCOUNTS` environment variable
2. Specify a comma-separated list of email addresses that are allowed to access the application

Example:
```bash
AUTHORIZED_ACCOUNTS=user1@yourcompany.com,user2@partnercompany.com
```

### Step 4: Configure Password Authentication

To enable the second layer of password authentication:

1. Set the following environment variables:
   ```bash
   AUTH_EMAIL=admin@example.com
   AUTH_PASSWORD=your_secure_password
   ```

2. You can set these in a `.env` file or as system environment variables.

### Step 5: Run the Application

```bash
streamlit run RAG.py
```

## Deployment

For production deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Environment Variables

Configuration options can be set using environment variables:

- `AUTHORIZED_DOMAINS`: Comma-separated list of authorized domains (can also be set in secrets.toml)
- `AUTHORIZED_ACCOUNTS`: Comma-separated list of authorized email addresses
- `AUTH_EMAIL`: Email for password authentication (second layer)
- `AUTH_PASSWORD`: Password for password authentication (second layer)
- Other configuration variables as documented in the original application

## Usage

1. Sign in with your Google account
2. If domain or account restrictions are configured, only authorized users will be able to proceed
3. Enter the application password when prompted
4. Use the interface to ingest PDF documents, review OCR results, and manage stored documents

## Implementation Details

See [GOOGLE_OAUTH_IMPLEMENTATION.md](GOOGLE_OAUTH_IMPLEMENTATION.md) for detailed information about how Google OAuth was implemented in this application.