package com.kennycason.ml.classifier.bayes

import com.kennycason.nlp.token.tokenizer.WhiteSpaceTokenizer
import com.kennycason.nlp.gram.Gram
import com.kennycason.nlp.gram.GramTokenizer
import com.kennycason.nlp.gram.NGramTokenizer
import com.kennycason.nlp.gram.SkipGramTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import com.kennycason.nlp.util.ResourceLoader
import org.eclipse.collections.impl.list.mutable.ListAdapter
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/17/16.
 */
class BayesianClassifierSkipGramVsNGramTest {
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/wikipedia"
    private val textTokenizer = EnglishTokenizer()
    private val ngramTokenizer = NGramTokenizer<String>(3)
    private val skipgramTokenizer = SkipGramTokenizer<String>(3,3)
    private val errorDelta = 0.05

    @Test
    fun test() {
        val ngramBackedClassifier = NaiveBayesianClassifier()
        ngramBackedClassifier.trainNegative(buildGrams("chemistry.txt", ngramTokenizer))
        ngramBackedClassifier.trainPositive(buildGrams("biology.txt", ngramTokenizer))
        ngramBackedClassifier.finalizeTraining()

        val skipgramBackedClassifier = NaiveBayesianClassifier()
        skipgramBackedClassifier.trainNegative(buildGrams("chemistry.txt", skipgramTokenizer))
        skipgramBackedClassifier.trainPositive(buildGrams("biology.txt", skipgramTokenizer))
        skipgramBackedClassifier.finalizeTraining()

        val ngramPositiveProbability = ngramBackedClassifier.classify(buildGrams("cell.txt", ngramTokenizer))
        val skipgramPositiveProbability = skipgramBackedClassifier.classify(buildGrams("cell.txt", skipgramTokenizer))

        // assert text is classified as biology for both models
        assertTrue(Math.abs(1.0f - ngramPositiveProbability) < errorDelta)
        assertTrue(Math.abs(1.0f - skipgramPositiveProbability) < errorDelta)
    }

    private fun buildGrams(resource: String, tokenizer: GramTokenizer<String>) = tokenizer.tokenize(
            StringTokenStream(
                    textTokenizer.tokenize(
                            ResourceLoader().toString("$resourcePath/$resource"))))

}