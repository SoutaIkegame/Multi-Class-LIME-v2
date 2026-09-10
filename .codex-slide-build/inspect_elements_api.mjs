import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
const s=p.slides.items[17];
console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(s.elements)),s.elements.items?.length);
for(const x of [s.shapes,s.images,s.tables,s.charts]) console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(x)),x.items?.length)
console.log('element',Object.getOwnPropertyNames(Object.getPrototypeOf(s.elements.items[0])))
