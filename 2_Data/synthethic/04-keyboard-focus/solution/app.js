const cards=[...document.querySelectorAll('.task-card')];
const drawer=document.querySelector('#drawer');
const closeBtn=document.querySelector('#close-drawer');
let lastFocused=null;
function openDrawer(source){lastFocused=source;drawer.hidden=false;drawer.classList.add('open');closeBtn.focus()}
function closeDrawer(){drawer.classList.remove('open');setTimeout(()=>{drawer.hidden=true;if(lastFocused)lastFocused.focus()},250)}
cards.forEach(card=>{
  card.addEventListener('click',()=>openDrawer(card));
  card.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openDrawer(card)}});
});
closeBtn.addEventListener('click',closeDrawer);
drawer.addEventListener('keydown',event=>{if(event.key==='Escape')closeDrawer()});
document.querySelectorAll('.filter-chip').forEach(chip=>chip.addEventListener('click',()=>{document.querySelectorAll('.filter-chip').forEach(c=>{c.classList.remove('active');c.setAttribute('aria-pressed','false')});chip.classList.add('active');chip.setAttribute('aria-pressed','true');const filter=chip.dataset.filter;cards.forEach(card=>card.hidden=filter==='mine'?card.dataset.owner!=='me':filter==='high'?card.dataset.priority!=='high':false)}));
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>{t.classList.remove('active');t.setAttribute('aria-selected','false')});tab.classList.add('active');tab.setAttribute('aria-selected','true')}));
document.querySelectorAll('.add-card,.fake-button,.column-menu').forEach(item=>item.addEventListener('click',()=>item.classList.toggle('selected')));
