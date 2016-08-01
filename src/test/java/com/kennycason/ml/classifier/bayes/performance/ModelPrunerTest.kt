package com.kennycason.ml.classifier.bayes.performance

import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.nlp.gram.NGramTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import org.junit.Test
import kotlin.test.assertEquals

/**
 * Created by kenny on 6/29/16.
 */
class ModelPrunerTest {

    @Test
    fun basicTest() {
        val ngramTokenizer = NGramTokenizer<String>(2)
        val textTokenizer = EnglishTokenizer()
        val classifer = NaiveBayesianClassifier()
        val modelPruner = ModelPruner(minFrequency = 2)
        classifer.trainPositive(ngramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("i like programming"))))
        classifer.trainPositive(ngramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("i like programming"))))
        classifer.trainPositive(ngramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("i whatev metroid"))))
        classifer.finalizeTraining()

        assertEquals(4, classifer.model.size)
        modelPruner.prune(classifer)
        assertEquals(2, classifer.model.size) // should prune away "i_whatev" and "whatev_metroid" as they only appear once (below min frequency)

        // now lets perform a prune of all terms that are near 0.5 probability (i.e. netural)
        classifer.trainNegative(ngramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("i like programming"))))
        classifer.trainNegative(ngramTokenizer.tokenize(StringTokenStream(textTokenizer.tokenize("i like programming"))))
        classifer.finalizeTraining()

        modelPruner.prune(classifer)
        assertEquals(0, classifer.model.size) // should prune away now remaining netural terms "i_like" and "like_programming"
    }
}