import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const source = process.argv[2];
const p = await PresentationFile.importPptx(await FileBlob.load(source));
console.log('slides', p.slides.items.length);
console.log('size', JSON.stringify(p.slideSize));
console.log('masters', p.masters.items.length);
console.log((await p.inspect({kind:'slide,layout,textbox,shape,chart,table', maxChars:30000})).ndjson);
for (const id of ['sh/9072xkry','sh/ozy1ofad','sh/cb2tkvap']) {
  try { const o=p.resolve(id); console.log('STYLE',id,JSON.stringify(o.text?.style)); } catch {}
}
