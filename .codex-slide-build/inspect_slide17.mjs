import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
console.log((await p.inspect({kind:'textbox,shape',target:{id:'sl/c7uhcr6t',beforeLines:0,afterLines:100},maxChars:12000})).ndjson)
