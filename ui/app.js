/* GFlowNet BioPoly Discovery — Enhanced Frontend */

// ═══ Expanded Polymer Database ═══
const POLYMER_DB = {
  'PET': {
    name: 'Polyethylene Terephthalate', abbr: 'PET', icon: '🧴', img: 'PET',
    smiles: 'O=C(c1ccc(C(=O)OCCO)cc1)OCCO',
    uses: ['Water bottles', 'Food containers', 'Polyester fibers'],
    tg: 70, tensile: 55, biodeg_years: 450,
    env_impact: 'Found in 94% of tap water samples. Takes 450+ years to decompose.',
    alternatives: [
      { rank:1, name:'BioEster-α (PLA-type)', smiles:'OC(C)C(=O)OC(C)C(=O)O', img:'pla_type', s_bio:.89, s_mech:.62, s_syn:.85, reward:.72, tg:57, tensile:50, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:900, desc:'Polylactic acid derivative — made from corn starch, industrially compostable' },
      { rank:2, name:'CycloLactone-β (PCL-type)', smiles:'O=C1CCCCCO1', img:'pcl_type', s_bio:.82, s_mech:.55, s_syn:.90, reward:.68, tg:-60, tensile:25, biodeg_months:12, category:'fast', polymerizable:true, mechanism:'Ring-Opening', improvement:450, desc:'Polycaprolactone — flexible, used in medical sutures' },
      { rank:3, name:'SucciPoly-γ (PBS-type)', smiles:'O=C(CCC(=O)OCCCCO)OCCCCO', img:'pbs_type', s_bio:.84, s_mech:.58, s_syn:.82, reward:.67, tg:-32, tensile:34, biodeg_months:8, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:675, desc:'Polybutylene succinate — soil-compostable mulch film alternative' },
      { rank:4, name:'HydroxyButy-δ (PHB-type)', smiles:'CC(O)CC(=O)O', img:'phb_type', s_bio:.87, s_mech:.48, s_syn:.78, reward:.63, tg:4, tensile:38, biodeg_months:4, category:'fast', polymerizable:true, mechanism:'Bio-fermentation', improvement:1350, desc:'Polyhydroxybutyrate — produced by bacteria, fully marine-degradable' },
      { rank:5, name:'GlycoEster-ε (PGA-type)', smiles:'OC(=O)COC(=O)CO', img:'pga_type', s_bio:.91, s_mech:.52, s_syn:.88, reward:.70, tg:36, tensile:62, biodeg_months:3, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:1800, desc:'Polyglycolic acid derivative — strongest biodegradable polyester' },
      { rank:6, name:'AdipoFlex-ζ (PBAT-type)', smiles:'O=C(CCCCC(=O)OCCCCO)O', img:'pbat_type', s_bio:.80, s_mech:.60, s_syn:.79, reward:.65, tg:-30, tensile:32, biodeg_months:10, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:540, desc:'Adipate copolyester — ecoflex-type, flexible compostable bags' },
    ]
  },
  'PE': {
    name: 'Polyethylene', abbr: 'PE (LDPE/HDPE)', icon: '🛍️', img: 'PE',
    smiles: 'CCCCCCCCCCCCCCCC', uses: ['Plastic bags', 'Bottles', 'Packaging film', 'Toys'],
    tg: -120, tensile: 28, biodeg_years: 500,
    env_impact: '~380 million tonnes produced yearly. Major ocean pollutant.',
    alternatives: [
      { rank:1, name:'BioFlex-α (PCL-blend)', smiles:'O=C(CCCCCO)O', img:'bio_flex', s_bio:.78, s_mech:.55, s_syn:.85, reward:.66, tg:-62, tensile:24, biodeg_months:12, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:500, desc:'Flexible like PE but biodegrades in soil within 1 year' },
      { rank:2, name:'StarchPoly-β', smiles:'OCC1OC(O)C(O)C(O)C1O', img:'starch', s_bio:.92, s_mech:.38, s_syn:.90, reward:.60, tg:-2, tensile:12, biodeg_months:3, category:'fast', polymerizable:true, mechanism:'Natural', improvement:2000, desc:'Thermoplastic starch — from potato/corn, home compostable' },
      { rank:3, name:'LactiWrap-γ (PLA)', smiles:'CC(O)C(=O)OC(C)C(=O)O', img:'pla_type', s_bio:.88, s_mech:.45, s_syn:.88, reward:.62, tg:55, tensile:50, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:1000, desc:'PLA film — transparent, good for food wrap' },
    ]
  },
  'PP': {
    name: 'Polypropylene', abbr: 'PP', icon: '📦', img: 'PP',
    smiles: 'CC(C)CC(C)CC(C)C', uses: ['Food packaging', 'Bottle caps', 'Automotive parts', 'Textiles'],
    tg: -10, tensile: 35, biodeg_years: 400,
    env_impact: '2nd most produced plastic. Non-recyclable when colored.',
    alternatives: [
      { rank:1, name:'RigidLact-α (L-Lactide)', smiles:'CC1OC(=O)C(C)OC1=O', img:'lactide', s_bio:.90, s_mech:.65, s_syn:.83, reward:.73, tg:97, tensile:55, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Ring-Opening', improvement:800, desc:'Rigid PLA from lactide — replaces PP in food containers' },
      { rank:2, name:'PHBStrong-β', smiles:'CC(O)CC(=O)O', img:'phb_type', s_bio:.87, s_mech:.58, s_syn:.80, reward:.68, tg:5, tensile:40, biodeg_months:5, category:'fast', polymerizable:true, mechanism:'Bio-fermentation', improvement:960, desc:'PHB variant — rigid enough for automotive interior trim' },
      { rank:3, name:'PBSRigid-γ', smiles:'O=C(CCC(=O)OCCCCO)OCCCCO', img:'pbs_type', s_bio:.83, s_mech:.60, s_syn:.82, reward:.67, tg:-32, tensile:36, biodeg_months:8, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:600, desc:'PBS — excellent heat resistance for microwave containers' },
    ]
  },
  'PS': {
    name: 'Polystyrene (Styrofoam)', abbr: 'PS', icon: '☕', img: 'PS',
    smiles: 'CC(c1ccccc1)CC(c1ccccc1)C', uses: ['Cups', 'Takeaway containers', 'Insulation', 'Packing peanuts'],
    tg: 100, tensile: 40, biodeg_years: 500,
    env_impact: 'Breaks into microplastics rapidly. Banned in 30+ countries.',
    alternatives: [
      { rank:1, name:'MoldedFiber-α', smiles:'OCC1OC(O)C(O)C(O)C1O', img:'starch', s_bio:.95, s_mech:.50, s_syn:.75, reward:.65, tg:80, tensile:15, biodeg_months:2, category:'fast', polymerizable:true, mechanism:'Natural', improvement:3000, desc:'Molded cellulose pulp — replaces styrofoam in food packaging' },
      { rank:2, name:'PLARigid-β', smiles:'OC(C)C(=O)OC(C)C(=O)O', img:'pla_type', s_bio:.88, s_mech:.62, s_syn:.85, reward:.72, tg:58, tensile:52, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:1000, desc:'Crystallized PLA — clear cups, deli containers' },
      { rank:3, name:'PHBVBlend-γ', smiles:'OC(CC(=O)O)CC', img:'phbv', s_bio:.86, s_mech:.55, s_syn:.80, reward:.66, tg:-5, tensile:25, biodeg_months:4, category:'fast', polymerizable:true, mechanism:'Bio-fermentation', improvement:1500, desc:'PHBV — marine-degradable, cutlery and utensils' },
    ]
  },
  'PVC': {
    name: 'Polyvinyl Chloride', abbr: 'PVC', icon: '🔧', img: 'PVC',
    smiles: 'CC(Cl)CC(Cl)CC(Cl)C', uses: ['Pipes', 'Window frames', 'Cable insulation', 'Flooring'],
    tg: 80, tensile: 50, biodeg_years: 1000,
    env_impact: 'Releases dioxins when burned. Contains phthalate plasticizers.',
    alternatives: [
      { rank:1, name:'BioPipe-α (PBS)', smiles:'O=C(CCC(=O)OCCCCO)OCCCCO', img:'pbs_type', s_bio:.84, s_mech:.65, s_syn:.82, reward:.71, tg:-32, tensile:36, biodeg_months:8, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:1500, desc:'PBS tubes for irrigation — soil-degradable after use' },
      { rank:2, name:'BioRigid-β (PLA)', smiles:'CC1OC(=O)C(C)OC1=O', img:'lactide', s_bio:.90, s_mech:.68, s_syn:.83, reward:.75, tg:97, tensile:55, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Ring-Opening', improvement:2000, desc:'High-crystallinity PLA — durable enough for profiles' },
    ]
  },
  'Nylon': {
    name: 'Nylon-6 (Polyamide 6)', abbr: 'Nylon', icon: '👕', img: 'Nylon',
    smiles: 'O=C1CCCCCN1', uses: ['Textiles', 'Carpets', 'Automotive gears', 'Fishing nets'],
    tg: 50, tensile: 70, biodeg_years: 200,
    env_impact: 'Ghost fishing nets kill 100,000+ marine animals/year.',
    alternatives: [
      { rank:1, name:'BioAmide-α', smiles:'NCC(=O)NCC(=O)O', img:'bio_amide', s_bio:.72, s_mech:.60, s_syn:.80, reward:.65, tg:80, tensile:45, biodeg_months:18, category:'moderate', polymerizable:true, mechanism:'Condensation', improvement:133, desc:'Bio-polyamide from castor oil — textile fiber replacement' },
      { rank:2, name:'SilkMimic-β', smiles:'NCC(=O)NCC(=O)NCC(=O)O', img:'silk_mimic', s_bio:.80, s_mech:.55, s_syn:.70, reward:.62, tg:100, tensile:50, biodeg_months:12, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:200, desc:'Recombinant silk protein — strong, marine-degradable fishing nets' },
    ]
  },
  'LDPE': {
    name: 'Low-Density Polyethylene', abbr: 'LDPE', icon: '🛒', img: 'LDPE',
    smiles: 'CCCCCCCCCC', uses: ['Squeezable bottles', 'Bread bags', 'Cling wrap', 'Garbage bags'],
    tg: -110, tensile: 15, biodeg_years: 500,
    env_impact: 'Most common single-use plastic in landfills.',
    alternatives: [
      { rank:1, name:'BioWrap-α (PBAT)', smiles:'O=C(CCCCC(=O)OCCCCO)O', img:'pbat_type', s_bio:.80, s_mech:.50, s_syn:.82, reward:.64, tg:-30, tensile:18, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:1000, desc:'PBAT cling film — certified compostable, same stretch as LDPE' },
      { rank:2, name:'CelluFilm-β', smiles:'OCC(O)C(O)C(O)C(O)CO', img:'cellu_film', s_bio:.90, s_mech:.35, s_syn:.85, reward:.58, tg:0, tensile:10, biodeg_months:2, category:'fast', polymerizable:true, mechanism:'Natural', improvement:3000, desc:'Regenerated cellulose (cellophane) — transparent, food-safe' },
    ]
  },
  'HDPE': {
    name: 'High-Density Polyethylene', abbr: 'HDPE', icon: '🧹', img: 'HDPE',
    smiles: 'CCCCCCCCCCCCCCCCCC', uses: ['Milk jugs', 'Detergent bottles', 'Pipes', 'Cutting boards'],
    tg: -90, tensile: 32, biodeg_years: 500,
    env_impact: 'Better recyclability than LDPE but still persists for centuries.',
    alternatives: [
      { rank:1, name:'StrongBio-α (PBS)', smiles:'O=C(CCC(=O)OCCCCO)OCCCCO', img:'pbs_type', s_bio:.84, s_mech:.60, s_syn:.82, reward:.68, tg:-32, tensile:34, biodeg_months:8, category:'fast', polymerizable:true, mechanism:'Condensation', improvement:750, desc:'PBS bottles — rigid enough for detergent, soil compostable' },
      { rank:2, name:'PHAMold-β', smiles:'CC(O)CC(=O)O', img:'phb_type', s_bio:.87, s_mech:.52, s_syn:.78, reward:.64, tg:5, tensile:30, biodeg_months:6, category:'fast', polymerizable:true, mechanism:'Bio-fermentation', improvement:1000, desc:'PHA injection molded — marine-degradable milk containers' },
    ]
  }
};

// Name aliases for search
const ALIASES = {
  'POLYETHYLENE TEREPHTHALATE':'PET', 'PLASTIC BOTTLE':'PET', 'WATER BOTTLE':'PET',
  'POLYETHYLENE':'PE', 'PLASTIC BAG':'PE', 'POLYPROPYLENE':'PP', 'BOTTLE CAP':'PP',
  'POLYSTYRENE':'PS', 'STYROFOAM':'PS', 'THERMOCOL':'PS', 'FOAM CUP':'PS',
  'POLYVINYL CHLORIDE':'PVC', 'VINYL':'PVC', 'PIPE':'PVC',
  'NYLON 6':'Nylon', 'NYLON-6':'Nylon', 'POLYAMIDE':'Nylon', 'FISHING NET':'Nylon',
  'LOW DENSITY POLYETHYLENE':'LDPE', 'CLING WRAP':'LDPE', 'GARBAGE BAG':'LDPE',
  'HIGH DENSITY POLYETHYLENE':'HDPE', 'MILK JUG':'HDPE', 'DETERGENT BOTTLE':'HDPE',
  'PLASTIC WRAP':'LDPE', 'FOOD CONTAINER':'PET', 'TAKEAWAY BOX':'PS', 'STRAW':'PP',
  'PLASTIC CUP':'PS', 'GROCERY BAG':'LDPE', 'TRASH BAG':'LDPE',
};

// ═══ Particle Canvas ═══
function initParticles() {
  const c = document.getElementById('particles-canvas'), ctx = c.getContext('2d');
  c.width = innerWidth; c.height = innerHeight;
  const pts = Array.from({length:50}, () => ({x:Math.random()*c.width, y:Math.random()*c.height, vx:(Math.random()-.5)*.3, vy:(Math.random()-.5)*.3, r:Math.random()*2+.5, o:Math.random()*.25+.1}));
  (function draw() {
    ctx.clearRect(0,0,c.width,c.height);
    pts.forEach((p,i)=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>c.width)p.vx*=-1; if(p.y<0||p.y>c.height)p.vy*=-1;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle=`rgba(16,185,129,${p.o})`; ctx.fill();
      pts.slice(i+1).forEach(q=>{const d=Math.hypot(p.x-q.x,p.y-q.y); if(d<140){ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.strokeStyle=`rgba(59,130,246,${.05*(1-d/140)})`;ctx.stroke();}});
    });
    requestAnimationFrame(draw);
  })();
  addEventListener('resize', ()=>{c.width=innerWidth;c.height=innerHeight});
}

// ═══ Molecule Image Fallback (If SVG is missing) ═══
function handleImageError(imgEl, smiles) {
  // If the SVG fails to load or isn't generated yet, fallback to SMILES text
  const w = imgEl.width || 300, h = imgEl.height || 190;
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `width:${w}px;height:${h}px;background:#0f172a;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:10px;text-align:center;box-sizing:border-box;border-radius:12px;border:1px solid rgba(255,255,255,0.05);`;
  wrapper.innerHTML = `
    <span style="color:#64748b;font-family:'JetBrains Mono',monospace;font-size:12px;margin-bottom:8px;word-break:break-all">${smiles}</span>
    <span style="color:#475569;font-size:10px">(Structure rendering unavailable)</span>
  `;
  imgEl.parentNode.replaceChild(wrapper, imgEl);
}

// ═══ Search ═══
function searchPolymer(query) {
  const q = query.trim().toUpperCase();
  // Direct match
  for (const [k, d] of Object.entries(POLYMER_DB)) {
    if (q === k || q === d.abbr?.toUpperCase() || d.name.toUpperCase().includes(q)) return { key: k, data: d };
  }
  // Alias match
  for (const [alias, key] of Object.entries(ALIASES)) {
    if (q.includes(alias) || alias.includes(q)) return { key, data: POLYMER_DB[key] };
  }
  return null;
}

function performSearch() {
  const input = document.getElementById('search-input').value;
  if (!input.trim()) return;
  document.getElementById('loading').classList.add('active');
  setTimeout(() => {
    const r = searchPolymer(input);
    document.getElementById('loading').classList.remove('active');
    if (r) { displayTarget(r.key, r.data); displayResults(r.data.alternatives); displayCharts(r.key, r.data); }
    else alert(`No data for "${input}". Try: PET, PE, PP, PS, PVC, Nylon, LDPE, HDPE, or common items like "water bottle", "plastic bag"`);
  }, 1000);
}

// ═══ Display Target ═══
function displayTarget(key, d) {
  const s = document.getElementById('target-section'); s.style.display = 'block';
  s.scrollIntoView({ behavior:'smooth', block:'start' });
  document.getElementById('target-name').textContent = `${d.icon} ${d.name} (${d.abbr || key})`;
  document.getElementById('target-uses').innerHTML = d.uses.map(u=>`<span class="quick-tag">${u}</span>`).join(' ');
  document.getElementById('target-tg').textContent = `${d.tg} °C`;
  document.getElementById('target-tensile').textContent = `${d.tensile} MPa`;
  document.getElementById('target-biodeg').textContent = `${d.biodeg_years} years`;
  document.getElementById('target-impact').textContent = d.env_impact;
  document.getElementById('target-smiles-text').textContent = d.smiles;
  
  const imgEl = document.getElementById('target-mol-img');
  imgEl.onerror = () => handleImageError(imgEl, d.smiles);
  imgEl.src = `mol_images/${d.img}.svg`;
}

// ═══ Display Results ═══
function displayResults(alts) {
  document.getElementById('results-section').style.display = 'block';
  const g = document.getElementById('results-grid'); g.innerHTML = '';
  alts.forEach((a, i) => {
    const cid = `mol-${i}`;
    const card = document.createElement('div');
    card.className = 'result-card'; card.style.animationDelay = `${i*.1}s`;
    card.style.cursor = 'pointer';
    card.onclick = () => openCandidateModal(a);
    card.innerHTML = `
      <div class="result-card-header">
        <div class="result-rank">${a.rank}</div>
        <div class="result-badges">
          <span class="badge badge-${a.category}">${a.category} biodeg</span>
          ${a.polymerizable?`<span class="badge badge-poly">${a.mechanism}</span>`:''}
        </div>
      </div>
      <div class="result-molecule"><img src="mol_images/${a.img}.svg" onerror="handleImageError(this, '${a.smiles}')" alt="Molecule Structure"></div>
      <div class="result-name">${a.name}</div>
      <div class="result-smiles">${a.smiles}</div>
      <p style="padding:0 20px 12px;font-size:13px;color:var(--text-secondary)">${a.desc||''}</p>
      <div class="result-scores">
        <div class="score-item"><div class="score-val" style="color:var(--accent-green)">${(a.s_bio*100).toFixed(0)}%</div><div class="score-label">Biodeg</div><div class="score-bar-container"><div class="score-bar bio" style="width:${a.s_bio*100}%"></div></div></div>
        <div class="score-item"><div class="score-val" style="color:var(--accent-blue)">${(a.s_mech*100).toFixed(0)}%</div><div class="score-label">Strength</div><div class="score-bar-container"><div class="score-bar mech" style="width:${a.s_mech*100}%"></div></div></div>
        <div class="score-item"><div class="score-val" style="color:var(--accent-purple)">${(a.s_syn*100).toFixed(0)}%</div><div class="score-label">Makeable</div><div class="score-bar-container"><div class="score-bar syn" style="width:${a.s_syn*100}%"></div></div></div>
      </div>
      <div class="result-footer"><span>Biodeg: ~${a.biodeg_months} months</span><span class="improvement">${a.improvement}× faster</span></div>`;
    g.appendChild(card);
  });
}

// ═══ Charts ═══
function displayCharts(key, d) {
  document.getElementById('charts-section').style.display = 'block';
  const alts = d.alternatives;

  // 1. Radar
  const rc = document.getElementById('radar-chart').getContext('2d');
  if (window._rc) window._rc.destroy();
  const colors = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#f43f5e','#06b6d4'];
  window._rc = new Chart(rc, { type:'radar', data:{
    labels:['Biodegradability','Mechanical Strength','Synthesizability','Overall Reward'],
    datasets: alts.slice(0,5).map((a,i)=>({label:a.name.split('(')[0].trim(),data:[a.s_bio,a.s_mech,a.s_syn,a.reward],borderColor:colors[i],backgroundColor:colors[i]+'18',borderWidth:2,pointRadius:4}))
  }, options:{scales:{r:{beginAtZero:true,max:1,grid:{color:'rgba(255,255,255,0.06)'},ticks:{display:false},pointLabels:{color:'#94a3b8',font:{size:12}}}},plugins:{legend:{labels:{color:'#94a3b8',font:{size:11}}}},responsive:true}});

  // 2. Biodeg time bar
  const bc = document.getElementById('bar-chart').getContext('2d');
  if (window._bc) window._bc.destroy();
  const items = [{n:key+' (current)',m:d.biodeg_years*12,c:'#f43f5e'},...alts.map(a=>({n:a.name.split('(')[0].trim(),m:a.biodeg_months,c:'#10b981'}))];
  window._bc = new Chart(bc, { type:'bar', data:{labels:items.map(i=>i.n),datasets:[{label:'Months',data:items.map(i=>Math.min(i.m,120)),backgroundColor:items.map(i=>i.c+'80'),borderColor:items.map(i=>i.c),borderWidth:1,borderRadius:6}]},
    options:{indexAxis:'y',scales:{x:{title:{display:true,text:'Months to biodegrade (capped at 120)',color:'#64748b'},grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#64748b'}},y:{grid:{display:false},ticks:{color:'#94a3b8',font:{size:11}}}},plugins:{legend:{display:false}},responsive:true}});

  // 3. Environmental impact doughnut
  const dc = document.getElementById('impact-chart').getContext('2d');
  if (window._dc) window._dc.destroy();
  const best = alts[0];
  const plasticYears = d.biodeg_years;
  const bioMonths = best.biodeg_months;
  window._dc = new Chart(dc, { type:'doughnut', data:{
    labels:['Time saved (years)','Remaining (months)'],
    datasets:[{data:[plasticYears, bioMonths/12], backgroundColor:['#10b98180','#f43f5e40'],borderColor:['#10b981','#f43f5e'],borderWidth:2}]
  }, options:{plugins:{legend:{labels:{color:'#94a3b8'}},tooltip:{callbacks:{label:ctx=>ctx.label+': '+ctx.parsed.toFixed(1)+' years'}}},responsive:true}});

  // 4. Mechanical comparison grouped bar
  const mc = document.getElementById('mech-chart').getContext('2d');
  if (window._mc) window._mc.destroy();
  const labels = [key+' (target)', ...alts.slice(0,4).map(a=>a.name.split(' ')[0])];
  window._mc = new Chart(mc, { type:'bar', data:{labels, datasets:[
    {label:'Tensile (MPa)',data:[d.tensile,...alts.slice(0,4).map(a=>a.tensile)],backgroundColor:'#3b82f660',borderColor:'#3b82f6',borderWidth:1,borderRadius:4},
    {label:'Tg (°C, +120)',data:[d.tg+120,...alts.slice(0,4).map(a=>a.tg+120)],backgroundColor:'#8b5cf660',borderColor:'#8b5cf6',borderWidth:1,borderRadius:4},
  ]}, options:{scales:{x:{grid:{display:false},ticks:{color:'#94a3b8'}},y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#64748b'}}},plugins:{legend:{labels:{color:'#94a3b8'}}},responsive:true}});
}

// ═══ Green AI Modal ═══
function openGreenAI() { document.getElementById('greenModal').classList.add('active'); }
function closeGreenAI() { document.getElementById('greenModal').classList.remove('active'); }

// ═══ Candidate Modal ═══
function openCandidateModal(a) {
  document.getElementById('cand-modal-name').textContent = a.name;
  document.getElementById('cand-modal-tg').textContent = `${a.tg} °C`;
  document.getElementById('cand-modal-tensile').textContent = `${a.tensile} MPa`;
  document.getElementById('cand-modal-biodeg').textContent = `~${a.biodeg_months} months`;
  document.getElementById('cand-modal-smiles').textContent = a.smiles;
  document.getElementById('cand-modal-img').src = `mol_images/${a.img}.svg`;
  document.getElementById('cand-modal-desc').textContent = a.desc || '';
  
  // Real-world feasibility analysis
  let feas = `This candidate has a synthesizability score of ${(a.s_syn*100).toFixed(0)}%. `;
  if (a.polymerizable) {
    feas += `It is highly feasible for physical manufacturing via <strong>${a.mechanism}</strong>. `;
  } else {
    feas += `It lacks known easy polymerization groups and might require complex synthetic pathways or functionalization prior to polymerization. `;
  }
  
  // Add use cases based on properties
  const useCases = [];
  if (a.tensile >= 40) useCases.push('structural components');
  if (a.tensile >= 20 && a.tensile < 40) useCases.push('flexible packaging');
  if (a.tg > 50) useCases.push('hot-fill containers');
  if (a.tg < 0) useCases.push('cold-chain packaging');
  if (a.biodeg_months <= 6) useCases.push('single-use food packaging');
  if (a.biodeg_months <= 3) useCases.push('compostable cutlery');
  
  if (useCases.length > 0) {
    feas += `<br><br><strong>Potential applications:</strong> ${useCases.join(', ')}.`;
  }
  
  document.getElementById('cand-modal-feasibility').innerHTML = feas;
  
  const badges = document.getElementById('cand-modal-badges');
  badges.innerHTML = `
    <span class="badge badge-${a.category}">${a.category} biodeg</span>
    ${a.polymerizable?`<span class="badge badge-poly">${a.mechanism}</span>`:''}
  `;
  document.getElementById('candidateModal').classList.add('active');
}
function closeCandidateModal() { document.getElementById('candidateModal').classList.remove('active'); }

function quickSearch(p) { document.getElementById('search-input').value = p; performSearch(); }

// ═══ Init ═══
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  document.getElementById('search-btn').addEventListener('click', performSearch);
  document.getElementById('search-input').addEventListener('keypress', e => { if(e.key==='Enter') performSearch(); });
  // Stat counters
  document.querySelectorAll('.stat-value[data-count]').forEach(el => {
    const t = parseInt(el.dataset.count); let c = 0; const s = Math.ceil(t/40);
    const timer = setInterval(()=>{c+=s;if(c>=t){c=t;clearInterval(timer)}el.textContent=c.toLocaleString()+(el.dataset.suffix||'')},30);
  });
  // Close modal on backdrop click
  document.querySelectorAll('.modal-overlay').forEach(el => {
    el.addEventListener('click', e => { 
      if(e.target.classList.contains('modal-overlay')) {
        closeGreenAI(); closeCandidateModal();
      }
    });
  });
  
  // Auto-load PET
  setTimeout(() => quickSearch('PET'), 500);
});
