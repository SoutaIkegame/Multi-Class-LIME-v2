import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
console.log('slides proto',Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides)));
console.log('slide proto',Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.items[0])));
