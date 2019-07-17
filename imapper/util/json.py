try:
    import ujson as json
except ImportError as e:
    print("[stealth/util/json.py] %s\nThis is fine, import json instead." % e)
    import json
# import json as json