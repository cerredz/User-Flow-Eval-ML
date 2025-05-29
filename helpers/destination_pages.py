# This function returns the desired destination pages for a singular page
# For example: the destination pages for '/home' map to 'login', 'pricing', or 'chat'
def get_desired_pages():
    destinations = {
        "/" : ["/login", "/chat", "/pricing"],
        "/chat": ["/chat/context", "/chat/questions", "/chat", "/pricing"],
        "/settings": ["/settings/subscription", "/"],
        "/pricing": ["/success"],
        "/login": ["/"]
    }

    return destinations


# This function returns the un-desired destination pages for a singular page
# For example: the destination pages for '/home' map to 'login', 'pricing', or 'chat'
def get_undesired_pages():
    destinations = { 
        "/settings": ["/settings/logout"],
        "/pricing": ["/blog", "contact", "/chat"],
        "/chat": ["/contact", "/settings"],
        "/": ["/contact"]
    }

    return destinations

