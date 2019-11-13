# encoding = utf-8
#

from document_retriver import bm25_retriver



def doc_Reader(context, query):
    '''
    Input: content paragraph and query question
    Output: answer texts
    '''
    context, query = refine(context), refine(query)
    
    #let's build a very raw reading comprehensive sentence retrive program
    sentences = [x + '。' for x in context.split() if x]

    q_ner = [x[2] for x in fool.analysis(query)[-1][0]]

    candidate_sents = [line for line in senteces if any([ner in line for ner in q_ner])] \
        if q_ner else sentences

    #tokenization
    candidate_sents = [list(cut(x)) for x in candidate_sents]

    #modeling
    retriver_idx = bm25_retriver(candidate_sents, query, mod = 'multi-sentences')

    res = candidate_sents[retriver_idx]

    return ''.join(res)



def run():
    data = pd.read_csv(config.train_file)

    for i in range(data.shape[0]):
        contents = [data['content1'][i], data['content2'][i], data['content3'][i], data['content4'][i], data['content5'][i]]
        query = data['question'][i]
        answer = re.sub(r'@(\w*)@', '', data['answer'][i])
        #target = re.findall("@(\w*)@", data['supporting_paragraph'][i])

        #multi-document retriver
        content_idx = bm25_retriver(contents, query, k=1, mod = 'multi-paragraphs')

        reader_output = doc_Reader(data[content_idx[0]], query)
        print('query:{}\n answer:{} \n reader_output:{}\n '.format(query, answer, reader_output))



if __name__ == '__main__':
    run()