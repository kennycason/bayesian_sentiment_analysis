package com.kennycason.ml.classifier.bayes.model

import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import com.kennycason.nlp.data.imdb.ImdbLoader
import com.kennycason.nlp.gram.GramTokenizer
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
class StochasticBayesianClassifierModelPersisterTest {
    private val resourcePath = "com/kennycason/ml/classifier/bayes/corpus/imdb_sample/"
    private val textTokenizer = EnglishTokenizer()
    private val gramTokenizer = NGramTokenizer<String>(2)
    private val modelPersister = StreamingModelPersister()
    private val modelFile = File(System.getProperty("java.io.tmpdir") + "/stochastic_naive_bayes.json")

    @Test
    fun persistAndLoad() {
        val classifier = StochasticBayesianClassifier(classifierCount = 3, samplingPercent = 0.2f)
        val imdbDataCorpus = ImdbLoader().loadFromResourceDirectory(resourcePath)

        classifier.trainPositive(buildGrams(imdbDataCorpus.train.positive))
        classifier.trainNegative(buildGrams(imdbDataCorpus.train.negative))
        classifier.finalizeTraining()

        val review1 = buildGrams(imdbDataCorpus.train.positive.first())
        val review2 = buildGrams(imdbDataCorpus.train.negative.first())

        // asserts pre-persist conditions
        val positiveProbability = classifier.classify(review1)
        val positiveProbability2 = classifier.classify(review2)

        modelPersister.persist(classifier, modelFile)

        // now load model and re-assert conditions
        val loadedClassifier = modelPersister.load(modelFile)
        assertTrue(loadedClassifier is StochasticBayesianClassifier)
        assertTrue(Math.abs(loadedClassifier.classify(review1) - positiveProbability) < 0.001)
        assertTrue(Math.abs(loadedClassifier.classify(review2) - positiveProbability2) < 0.001)
    }

    private fun buildGrams(data: String) =
                gramTokenizer.tokenize(StringTokenStream(
                        textTokenizer.tokenize(data)))

    private fun buildGrams(corpus: List<String>) =
            corpus.map { data ->
                gramTokenizer.tokenize(StringTokenStream(
                        textTokenizer.tokenize(data))) }
}
