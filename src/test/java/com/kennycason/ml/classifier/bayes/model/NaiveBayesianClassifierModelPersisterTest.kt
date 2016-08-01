package com.kennycason.ml.classifier.bayes.model

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.nlp.gram.NGramTokenizer
import com.kennycason.nlp.token.StringTokenStream
import com.kennycason.nlp.token.tokenizer.EnglishTokenizer
import com.kennycason.nlp.util.ResourceLoader
import org.eclipse.collections.impl.list.mutable.ListAdapter
import org.junit.Test
import java.io.File
import kotlin.test.assertTrue

/**
 * Created by kenny on 6/20/16.
 */
class NaiveBayesianClassifierModelPersisterTest {
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/email"
    private val textTokenizer = EnglishTokenizer()
    private val ngramTokenizer = NGramTokenizer<String>(2)
    private val modelPersister = StreamingModelPersister()
    private val modelFile = File(System.getProperty("java.io.tmpdir") + "/naive_bayes.json")

    @Test
    fun persistAndLoad() {
        val classifier = NaiveBayesianClassifier()
        classifier.trainNegative(buildGrams("spam.txt"))
        classifier.trainPositive(buildGrams("good.txt"))
        classifier.finalizeTraining()

        // asserts pre-persist conditions
        assertModelValid(classifier)

        modelPersister.persist(classifier, modelFile)

        // now load model and re-assert conditions
        val loadedClassifier = modelPersister.load(modelFile)
        assertTrue(loadedClassifier is NaiveBayesianClassifier)
        assertModelValid(loadedClassifier)
    }

    private fun assertModelValid(classifier: BayesianClassifier) {
        val positiveProbability = classifier.classify(buildGrams("mail1.txt"))
        assertTrue(positiveProbability > 0.99)
        val positiveProbability3 = classifier.classify(buildGrams("mail2.txt"))
        assertTrue(positiveProbability3 < 0.01)
    }

    private fun buildGrams(resource: String) =
            ngramTokenizer.tokenize(
                StringTokenStream(textTokenizer.tokenize(
                        ResourceLoader().toString("$resourcePath/$resource"))))
}
