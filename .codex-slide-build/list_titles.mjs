import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
const x=await p.inspect({kind:'slide',maxChars:10000});
for(const l of x.ndjson.trim().split('\n')){const o=JSON.parse(l); console.log(o.slide+'\t'+o.id+'\t'+o.title)}
