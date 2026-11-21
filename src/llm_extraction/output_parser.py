"""Strict JSON structure with logged partial schema/type failures."""
import json
import re


def parse_response(text,allowed):
    result=dict(attributes=[],json_valid=False,structure_valid=False,issues=[],fence_removed=False,generated_items=0)
    body=text.strip()
    match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',body)
    if match:
        body=match[1];result['fence_removed']=True
    def unique_keys(items):
        output={}
        for key,value in items:
            if key in output:
                raise ValueError('Duplicate JSON key')
            output[key]=value
        return output
    try:
        obj=json.loads(body,object_pairs_hook=unique_keys)
    except (json.JSONDecodeError,ValueError) as error:
        result['issues'].append(dict(type='invalid_json',message=str(error)))
        return result
    result['json_valid']=True
    if not isinstance(obj,dict) or set(obj)!={'attributes'} or not isinstance(obj['attributes'],list) or len(obj['attributes'])>100:
        result['issues'].append(dict(type='invalid_structure'))
        return result
    result['structure_valid']=True
    result['generated_items']=len(obj['attributes'])
    for item in obj['attributes']:
        if not isinstance(item,dict) or set(item)-{'attribute','value','source'} or not {'attribute','value'}<=set(item):
            result['issues'].append(dict(type='invalid_item'));continue
        if not isinstance(item['attribute'],str) or item['attribute'] not in allowed:
            result['issues'].append(dict(type='out_of_schema',item=item));continue
        if not isinstance(item['value'],str) or not item['value'] or len(item['value'])>1000 or ('source' in item and not isinstance(item['source'],str)):
            result['issues'].append(dict(type='invalid_value',item=item));continue
        result['attributes'].append(item)
    return result
