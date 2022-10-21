from sentence_transformers import SentenceTransformer, util
import spacy
import scipy
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)
paragrah_seg = spacy.load("en_core_web_sm")


def segment_paragraph(paragraph):
    doc = paragrah_seg(paragraph)
    return [sentence.text.strip() for sentence in doc.sents]


def semantic_senetence_similarity(s1, s2):
    embeddings1 = model.encode(s1)
    embeddings2 = model.encode(s2)

    # Compute cosine-similarits
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    # print(f"Simliarity b/w '{s1}' & '{s2}' is {cosine_score[0][0]:.2f}")
    return cosine_score[0][0]


def match_answer_with_key(answer, key, threshold=0.5):
    segmented_answer_sentences = segment_paragraph(answer)
    segmented_key_sentences = segment_paragraph(key)
    sentence_covereded_in_answer = [False for _ in range(len(segmented_key_sentences))]
    matched_sentences = []
    for answer_sentence in segmented_answer_sentences:
        for key_sentence in segmented_key_sentences:
            similarity = semantic_senetence_similarity(answer_sentence, key_sentence)
            if similarity > threshold:
                sentence_covereded_in_answer[segmented_key_sentences.index(key_sentence)] = True
                obj = {
                    "answer_sentence": answer_sentence,
                    "key_sentence": key_sentence,
                    "similarity_score": similarity.item(),
                }
                matched_sentences.append(obj)

    # filter key sentences not in answer
    key_sentences_not_in_answer = [
        segmented_key_sentences[i]
        for i in range(len(segmented_key_sentences))
        if sentence_covereded_in_answer[i] == False and len(segmented_key_sentences[i]) != 0
    ]
    key_sentences_in_answer = [
        segmented_key_sentences[i]
        for i in range(len(segmented_key_sentences))
        if sentence_covereded_in_answer[i] == True and len(segmented_key_sentences[i]) != 0
    ]
    # print(key_sentences_not_in_answer)

    # calculate semantic score
    total_len = sum(len(sent) for sent in segmented_key_sentences)
    sum_ = sum(
        len(segmented_key_sentences[i])
        for i in range(len(segmented_key_sentences))
        if sentence_covereded_in_answer[i] == True
    )

    final_score = 10 * sum_ / total_len
    # json_object = json.dumps(final_dict)
    # print(f'Total: {total_len}; Sum: {sum_}; Final_Score: {final_score}')
    return {
        "semantic_score": final_score,
        "key_sentences_not_in_answer": key_sentences_not_in_answer,
        "key_sentences_in_answer": key_sentences_in_answer,
        "matched_sentences": matched_sentences,
    }
    # print(f"Simlilar sentences found! \n S1: {answer_sentence} \n S2: {key_sentence} \n With Similarity Score: {similarity:.2f}")


if __name__ == "__main__":
    # Sentence semantic similarity
    s1 = "People need to log off their computers."
    s2 = "It is necessary for people to log off from their computers."
    score = semantic_senetence_similarity(s1, s2)
    print(f"The semantic similarity score is {score:.2f}")

    sample_key = "Cat is a very adorable and a cute animal. It is a domestic animal and is kept as a pet. It has very sharp claws and keen eyes that help it in seeing during the night. That means that it has a very good nocturnal vision that is much better than humans. It has two small ears, with a highly sensitive tympanic membrane (eardrum), which helps it in hearing even the slightest of the sounds. It’s small and bushy tail helps it in maintaining balance while walking.Cat is an extremely beautiful and a mesmerising mammal, which can attract you towards itself with it’s laid back attitude and funny portrayal of it’s actions. You will be completely fascinated by the cat. It can be aggressive at times, when it is irritated or is being continuously poked. Cats are found in many colours like brown, golden, white, black or a mix of any these two colours."

    segmented_key_sentences = segment_paragraph(sample_key)
    print("Segmented paragraph for key is")
    print(segmented_key_sentences)

    sample_answer = """The cat is a really lovely and adorable animal. It is a pet that is kept as a domestic animal. It possesses razor-sharp claws and strong eyes that aid it in night vision. That implies it has excellent nocturnal eyesight, far superior to that of humans.
    # It has two tiny ears and an extremely sensitive tympanic membrane (eardrum) that allows it to hear even the smallest noises. It walks with a tiny, bushy tail that helps it maintain balance.Cat is a mesmerising and incredibly attractive animal that may draw you in with its laid-back demeanour and amusing representation of its behaviours. The cat will hold your attention totally."""

    segmented_answer_sentences = segment_paragraph(sample_answer)
    print("Segmented paragraph for sample answer is")
    print(segmented_answer_sentences)

    match_answer_with_key(sample_answer, sample_key)
