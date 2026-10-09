"""Small migration checks: exclusion, redaction, checksum and safe restore."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class HandoffTest(unittest.TestCase):
 def test_export_restore(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);src=root/'src';dst=root/'dst';src.mkdir();(src/'training_runs/run').mkdir(parents=True)
   secret='sk-'+'x'*32
   (src/'.env').write_text('API_KEY='+secret)
   run=src/'training_runs/run';(run/'result.json').write_text(json.dumps({'answer':'A','credential':secret}))
   (run/'video.mp4').write_bytes(b'video');(run/'model.safetensors').write_bytes(b'weight')
   subprocess.run([sys.executable,str(HERE/'export_snapshot.py'),'--source',str(src),'--dest',str(dst)],check=True,stdout=subprocess.DEVNULL)
   subprocess.run([sys.executable,str(HERE/'restore_bundles.py'),'--repo',str(dst)],check=True,stdout=subprocess.DEVNULL)
   self.assertNotIn(secret,(dst/'training_runs/run/result.json').read_text());self.assertFalse((dst/'.env').exists());self.assertFalse((dst/'training_runs/run/video.mp4').exists());self.assertFalse((dst/'training_runs/run/model.safetensors').exists())
   m=json.loads((dst/'handoff/bundles/manifest.json').read_text());part=dst/'handoff/bundles'/m[0]['parts'][0]['file'];part.write_bytes(b'corrupt')
   p=subprocess.run([sys.executable,str(HERE/'restore_bundles.py'),'--repo',str(dst),'--verify-only'],capture_output=True)
   self.assertNotEqual(p.returncode,0)
if __name__=='__main__':unittest.main()
