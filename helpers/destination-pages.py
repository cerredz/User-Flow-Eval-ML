# This function returns the desired destination pages for a singular page
# For example: the destination pages for '/home' map to 'login', 'pricing', or 'chat'
def get_desired_pages():
    destinations = {
        "/home" : ["/login", "/chat", "/pricing"],
        "/chat": ["/chat/context", "/chat/questions"],
        "/settings": ["/settings/subscription", "/home"],
        "/pricing": ["/success"]
    }

    return destinations


# This function returns the un-desired destination pages for a singular page
# For example: the destination pages for '/home' map to 'login', 'pricing', or 'chat'
def get_undesired_pages():
    destinations = { 
        "/settings": ["/settings/logout"],
        "/pricing": ["/blog", "contact"],
        "/chat": ["/contact"]
    }

    return destinations