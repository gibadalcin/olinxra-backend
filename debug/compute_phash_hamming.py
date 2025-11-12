from PIL import Image
import imagehash, json, sys
r=json.load(open('debug/preproc/report.json'))
cand_path=r.get('candidate')
print('candidate path:',cand_path)
try:
    cand=Image.open(cand_path).convert('RGB')
    ph_c=str(imagehash.phash(cand))
except Exception as e:
    ph_c=None
    print('failed to compute candidate phash:',e)
print('candidate phash:',ph_c)
for res in r['results']:
    ph=res['phash']
    try:
        h=imagehash.hex_to_hash(ph) - imagehash.hex_to_hash(ph_c)
    except Exception as e:
        h='ERR'
    print(res['variant'],res['mode'], 'phash',ph,'hamming',h,'mse',res['mse_thumb'])
