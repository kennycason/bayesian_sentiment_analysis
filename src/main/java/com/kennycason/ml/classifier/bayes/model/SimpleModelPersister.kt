package com.kennycason.ml.classifier.bayes.model

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import org.apache.commons.io.IOUtils
import org.eclipse.collections.impl.factory.Lists
import java.io.File
import java.io.FileReader
import java.io.FileWriter

/**
 * Created by kenny on 6/20/16.
 *
 * Loading all the training data and training a Bayesian classifier(s) is costly. As such for our core classifier
 * we can persist a trained model, and thus load a trained model into memory and skip the training phase.
 * The model is persisted in JSON.
 *
 * This class loads EVERYTHING into memory and while easier to maintain code-wise, it isn't memory friendly.
 * NOTE the format is now divergent form the StreamingModelPersister
 *
 * Format Sample
 *  {
 *       "meta" : {
 *           "samplingPercent" : 0.2,
 *           "exclusions" : ["foo", "bar"],
 *           "interestingGramsCount" : 15,
 *           "assumePrioriWhenSubjectAbsent" : true,
 *           "negativeProbabilityPriori" : 0.4
 *       },
 *       "models" : [
 *          [
 *              {
 *                  "token" : "best_known",
 *                  "negativeCount" : 0,
 *                  "positiveCount" : 1,
 *                  "negativeRatio" : 0.0,
 *                  "positiveRatio" : 0.004048583,
 *                  "positiveProbability" : 0.99,
 *                  "negativeProbability" : 0.01
 *              },
 *              {...}
 *          ],
 *          [other models]
 *      ]
 *  }
 *
 * Format Sample for NaiveBayesianClassifier
 *  {
 *       "meta" : {
 *           "exclusions" : ["foo", "bar"],
 *           "interestingGramsCount" : 15,
 *           "assumePrioriWhenSubjectAbsent" : true,
 *           "negativeProbabilityPriori" : 0.4
 *       },
 *       "model" : [
 *           {
 *               "token" : "best_known",
 *               "negativeCount" : 0,
 *               "positiveCount" : 1,
 *               "negativeRatio" : 0.0,
 *               "positiveRatio" : 0.004048583,
 *               "positiveProbability" : 0.99,
 *               "negativeProbability" : 0.01
 *           },
 *           {...}
 *       ]
 *   }
 */
@Deprecated("Deprecated for now, still experimental")
class SimpleModelPersister {
    private val objectMapper = ObjectMapper()

    fun persist(classifier: NaiveBayesianClassifier, outputFile: File) {
        val json = objectMapper
                .writerWithDefaultPrettyPrinter()
                .writeValueAsString(Model(buildMeta(classifier), buildModels(classifier)))
        val fileWriter = FileWriter(outputFile)
        IOUtils.write(json, fileWriter)
        fileWriter.close()
    }

    fun persist(classifier: StochasticBayesianClassifier, outputFile: File) {
        val json = objectMapper
                .writerWithDefaultPrettyPrinter()
                .writeValueAsString(Model(buildMeta(classifier), buildModels(classifier.classifiers)))
        val fileWriter = FileWriter(outputFile)
        IOUtils.write(json, fileWriter)
        fileWriter.close()
    }

    fun load(inputFile: File): BayesianClassifier {
        val model = objectMapper.readValue(FileReader(inputFile), Model::class.java)
        return if (model.models.size < 1) {
            throw IllegalArgumentException("Loaded classifier must have at least one model.")

        } else if (model.models.size == 1) {
            buildSingleClassifier(model.meta, model.models.first())

        } else {
            buildStochasticClassifier(model)
        }
    }

    private fun buildStochasticClassifier(model: Model): StochasticBayesianClassifier {
        val classifiers = mutableListOf<NaiveBayesianClassifier>()
        model.models.forEach { subjects ->
            classifiers.add(buildSingleClassifier(model.meta, subjects))
        }
        return StochasticBayesianClassifier(
                samplingPercent = model.meta.samplingPercent,
                exclusions = model.meta.exclusions,
                interestingGramsCount = model.meta.interestingGramsCount,
                assumePrioriWhenSubjectAbsent = model.meta.assumePrioriWhenSubjectAbsent,
                negativeProbabilityPriori = model.meta.negativeProbabilityPriori,
                preBuiltClassifiers = classifiers)
    }

    private fun buildSingleClassifier(meta: Meta, subjects: List<NaiveBayesianClassifier.Subject>): NaiveBayesianClassifier {
        val classifier = NaiveBayesianClassifier(
                exclusions = meta.exclusions,
                interestingGramsCount = meta.interestingGramsCount,
                assumePrioriWhenSubjectAbsent = meta.assumePrioriWhenSubjectAbsent,
                negativeProbabilityPriori = meta.negativeProbabilityPriori)
        classifier.model.putAll(subjects.associateBy({ s -> s.token}))
        return classifier
    }

    private fun buildModels(classifiers: List<NaiveBayesianClassifier>): MutableList<MutableList<NaiveBayesianClassifier.Subject>> {
        val models: MutableList<MutableList<NaiveBayesianClassifier.Subject>> = mutableListOf()
        classifiers.forEach { classifier ->
            models.add(classifier.model.values.toMutableList())
        }
        return models
    }

    private fun buildModels(classifier: NaiveBayesianClassifier): MutableList<MutableList<NaiveBayesianClassifier.Subject>> {
        val models: MutableList<MutableList<NaiveBayesianClassifier.Subject>> = mutableListOf()
        models.add(classifier.model.values.toMutableList())
        return models
    }

    private fun buildMeta(classifier: StochasticBayesianClassifier): Meta {
        val meta = buildMeta(classifier as BayesianClassifier)
        meta.samplingPercent = classifier.samplingPercent
        return meta
    }

    private fun buildMeta(classifier: BayesianClassifier) = Meta(
            exclusions = classifier.exclusions,
            interestingGramsCount = classifier.interestingGramsCount,
            assumePrioriWhenSubjectAbsent = classifier.assumePrioriWhenSubjectAbsent,
            negativeProbabilityPriori = classifier.negativeProbabilityPriori)

    data class Model(var meta: Meta = Meta(),
                     var models: MutableList<MutableList<NaiveBayesianClassifier.Subject>> = mutableListOf())

    data class Meta(var samplingPercent: Float = 0.2f,
                    var exclusions: MutableSet<String> = mutableSetOf(),
                    var interestingGramsCount: Int = 15,
                    var assumePrioriWhenSubjectAbsent: Boolean = false,
                    var negativeProbabilityPriori: Float = 0.4f)
}