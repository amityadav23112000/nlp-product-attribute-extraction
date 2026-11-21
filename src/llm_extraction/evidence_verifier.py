"""Retain exact source occurrences and distinguish repeated/unsupported evidence."""
def verify_attributes(attributes,product):
    all_predictions=[];accepted=[]
    for item in attributes:
        matches=[]
        for source in product['sources']:
            if 'source' in item and item['source']!=source['source']:
                continue
            text=source['text'];start=0
            while True:
                start=text.find(item['value'],start)
                if start<0:break
                matches.append(dict(source=source['source'],start=start,end=start+len(item['value']),text=text[start:start+len(item['value'])]))
                start+=1
        status='exact' if len(matches)==1 else 'ambiguous' if matches else 'unsupported'
        prediction=dict(item,evidence_status=status,evidence=matches)
        all_predictions.append(prediction)
        if matches:accepted.append(prediction)
    return all_predictions,accepted
