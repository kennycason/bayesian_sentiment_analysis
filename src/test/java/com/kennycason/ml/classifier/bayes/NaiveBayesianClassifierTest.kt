package com.kennycason.ml.classifier.bayes

import com.kennycason.nlp.token.tokenizer.WhiteSpaceTokenizer
import com.kennycason.nlp.gram.Gram
import com.kennycason.nlp.gram.NGramTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import com.kennycason.nlp.util.ResourceLoader
import org.eclipse.collections.impl.list.mutable.ListAdapter
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/15/16.
 */
class NaiveBayesianClassifierTest {
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/email"
    private val textTokenizer = EnglishTokenizer()
    private val ngramTokenizer = NGramTokenizer<String>(2)
    private val errorDelta = 0.05

    @Test
    fun test() {
        val classifier = NaiveBayesianClassifier()

        // Train spam with a file of spam e-mails
        classifier.trainNegative(buildGrams("spam.txt"))

        // Train spam with a file of regular e-mails
        classifier.trainPositive(buildGrams("good.txt"))

        // We are finished adding words so finalize the results
        classifier.finalizeTraining()

        // train data
        val positiveProbabilityTrain = classifier.classify(buildGrams("good.txt"))
        assertTrue(positiveProbabilityTrain > 0.99)

        val negativeProbabilityTrain = classifier.classify(buildGrams("spam.txt"))
        assertTrue(negativeProbabilityTrain < 0.01)

        // test good
        val positiveProbability = classifier.classify(buildGrams("mail1.txt"))
        assertTrue(positiveProbability > 0.9)

        // test bad
        val positiveProbability3 = classifier.classify(buildGrams("mail2.txt"))
        assertTrue(positiveProbability3 < 0.01)
    }

    private fun buildGrams(resource: String) = ngramTokenizer.tokenize(
            StringTokenStream(
                    textTokenizer.tokenize(
                            ResourceLoader().toString("$resourcePath/$resource"))))
                                    .filter { token -> token.buildToken().length > 4 }

}