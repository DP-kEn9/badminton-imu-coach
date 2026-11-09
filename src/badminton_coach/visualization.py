import json
from pathlib import Path

from .body import PARENTS


def replay(path, pose, rate, truth=None):
    data = {"prediction": pose.tolist(), "parents": PARENTS, "rate": rate}
    if truth is not None:
        data["truth"] = truth.tolist()
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>IMU 相对姿态回放</title>
<style>body{background:#101c2c;color:#e5edfa;font:16px system-ui;margin:28px}canvas{background:#15253a;max-width:100%;border-radius:12px}button,input{margin:12px}small{display:block;color:#bbcadd}</style>
<h1>四身体 IMU · 骨盆相对姿态</h1><p>synthetic_analytic / synthetic_demo_only · fused_orientation</p>
<small>蓝色：训练模型预测。无绝对场地轨迹、根高度和球拍。动画自然不代表重建准确。</small>
<canvas id="c" width="900" height="550"></canvas><br><button id="play">暂停</button>
<input id="time" type="range" min="0" value="0" style="width:55%"><span id="label"></span>
<label id="gtlabel" hidden><input id="gt" type="checkbox">研发评测 GT（橙色）</label>
<script>const D=DATA; const c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('time');
s.max=D.prediction.length-1;let playing=true,index=0,last=performance.now();
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};
s.oninput=()=>{index=+s.value;draw()}; if(D.truth)document.getElementById('gtlabel').hidden=false;
function project(p){return [450+220*(p[1]+.45*p[0]),260-220*p[2]+60*p[0]]}
function skeleton(points,color){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=4;points.forEach((p,j)=>{let a=project(p);if(D.parents[j]>=0){let b=project(points[D.parents[j]]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.beginPath();ctx.arc(...a,4,0,Math.PI*2);ctx.fill()})}
function draw(){ctx.clearRect(0,0,900,550);skeleton(D.prediction[index],'#58bbff');if(D.truth&&document.getElementById('gt').checked)skeleton(D.truth[index],'#ffaa55');document.getElementById('label').textContent=(index/D.rate).toFixed(2)+' s';s.value=index}
function tick(now){if(playing&&now-last>=1000/D.rate){index=(index+Math.floor((now-last)*D.rate/1000))%D.prediction.length;last=now;draw()}else if(!playing)last=now;requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);</script></html>"""
    Path(path).write_text(html.replace("DATA", json.dumps(data)), encoding="utf-8")
