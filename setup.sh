mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your_heroku@email_id.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor="#031775"\n\
backgroundColor="#FFFFFF"\n\
secondaryBackgroundColor="#8D8FA5"\n\
textColor="#242424"\n\
font=\"Custom Pret à dépenser\"\n\
" > ~/.streamlit/config.toml