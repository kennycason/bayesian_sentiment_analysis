package com.kennycason.ml.classifier.bayes.model

import com.fasterxml.jackson.core.*
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.kennycason.ml.classifier.bayes.BayesianClassifier
import com.kennycason.ml.classifier.bayes.NaiveBayesianClassifier
import com.kennycason.ml.classifier.bayes.StochasticBayesianClassifier
import com.kennycason.nlp.util.ResourceLoader
import org.apache.commons.io.IOUtils
import org.eclipse.collections.impl.factory.Lists
import java.io.File
import java.io.FileReader
import java.io.FileWriter
import java.io.InputStream

/**
 * Created by kenny on 6/27/16.
 *
 * Loading all the training data and training a Bayesian classifier(s) is costly. As such for our core classifier
 * we can persist a trained model, and thus load a trained model into memory and skip the training phase.
 * The model is persisted in JSON.
 *
 * Use streaming read/write for performance since the resulting Json files are very large.
 *
 * Sorry for anyone who has to maintain this. I'll make this better :)
 *
 * Format Sample for StochasticBayesianClassifier
 *  {
 *       "meta" : {
 *           "samplingPercent" : 0.2,
 *           "exclusions" : ["foo", "bar"],
 *           "interestingGramsCount" : 15,
 *           "assumePrioriWhenSubjectAbsent" : true,
 *           "negativeProbabilityPriori" : 0.4
 *       },
 *       "classifiers" : [ NaiveBayesianClassifier ]
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
class StreamingModelPersister {
    private val jsonFactory = JsonFactory()
    init {
        jsonFactory.codec = ObjectMapper()
    }

    fun persist(classifier: NaiveBayesianClassifier, outputFile: File) {
        val jsonGenerator = jsonFactory.createGenerator(outputFile, JsonEncoding.UTF8)
        jsonGenerator.setPrettyPrinter(DefaultPrettyPrinter())
        writeClassifier(classifier, jsonGenerator)
        jsonGenerator.close()
    }

    fun persist(classifier: StochasticBayesianClassifier, outputFile: File) {
        val jsonGenerator = jsonFactory.createGenerator(outputFile, JsonEncoding.UTF8)
        jsonGenerator.setPrettyPrinter(DefaultPrettyPrinter())
        jsonGenerator.writeStartObject()
        writeMeta(jsonGenerator, buildMeta(classifier))

        jsonGenerator.writeArrayFieldStart("classifiers")
        classifier.classifiers.forEach { classifier ->
            writeClassifier(classifier, jsonGenerator)
        }
        jsonGenerator.writeEndArray()
        jsonGenerator.writeEndObject()
        jsonGenerator.close()
    }

    fun loadResource(resource: String): BayesianClassifier {
        return load(jsonFactory.createParser(ResourceLoader().toInputStream(resource)))
    }

    fun load(inputFile: File): BayesianClassifier {
        return load(jsonFactory.createParser(inputFile))
    }

    fun load(inputStream: InputStream): BayesianClassifier {
        return load(jsonFactory.createParser(inputStream))
    }

    private fun load(jsonParser: JsonParser): BayesianClassifier {
        jsonParser.nextToken() // consume first {

        var meta = readMeta(jsonParser)
        when (meta.model) {
            NaiveBayesianClassifier::class.java.simpleName -> return readSingleClassifier(meta, jsonParser)
            StochasticBayesianClassifier::class.java.simpleName -> return readStochasticClassifier(meta, jsonParser)
        }
        throw IllegalStateException("Encountered unknown model format: ${meta.model}")
    }

    /*
        read helpers
     */
    private fun readMeta(jsonParser: JsonParser): Meta {
        val meta = Meta()
        jsonParser.nextToken() // consume "{"
        jsonParser.nextToken() // consume "meta"
        jsonParser.nextToken() // consume "{"
        while (jsonParser.currentToken != JsonToken.END_OBJECT) {
            val currentName = jsonParser.currentName
            jsonParser.nextToken() // consume field name
            when (currentName) {
                "model" -> meta.model = jsonParser.text
                "samplingPercent" -> meta.samplingPercent = jsonParser.floatValue
                "interestingGramsCount" -> meta.interestingGramsCount = jsonParser.intValue
                "assumePrioriWhenSubjectAbsent" -> meta.assumePrioriWhenSubjectAbsent = jsonParser.booleanValue
                "negativeProbabilityPriori" -> meta.negativeProbabilityPriori = jsonParser.floatValue
                "exclusions" -> {
                    jsonParser.nextToken() // consume "["
                    if (jsonParser.currentToken != JsonToken.END_ARRAY) {
                        while (jsonParser.nextToken() != JsonToken.END_ARRAY) {
                            meta.exclusions.add(jsonParser.text)
                        }
                    }
                }
            }
            jsonParser.nextToken() // consume value or ending "]" of exclusions
        }
        jsonParser.nextToken() // consume  "}"

        return meta
    }

    // assume the meta data has already been read and build the remainder of the single classifier
    private fun readSingleClassifier(meta: Meta, jsonParser: JsonParser): NaiveBayesianClassifier {
        val classifier = NaiveBayesianClassifier(
                exclusions = meta.exclusions,
                interestingGramsCount = meta.interestingGramsCount,
                assumePrioriWhenSubjectAbsent = meta.assumePrioriWhenSubjectAbsent,
                negativeProbabilityPriori = meta.negativeProbabilityPriori)

        if (jsonParser.currentName == "model") {
            jsonParser.nextToken() // consume "["
            jsonParser.nextToken() // consume "{"
            while (jsonParser.currentToken != JsonToken.END_ARRAY) {
                val subject = NaiveBayesianClassifier.Subject()

                jsonParser.nextToken() // consume "{"
                while (jsonParser.currentToken != JsonToken.END_OBJECT) {
                    val currentName = jsonParser.currentName
                    jsonParser.nextToken() // consume value
                    when (currentName) {
                        "token" -> subject.token = jsonParser.text
                        "negativeCount" -> subject.negativeCount = jsonParser.intValue
                        "positiveCount" -> subject.positiveCount = jsonParser.intValue
                        "negativeRatio" -> subject.negativeRatio = jsonParser.floatValue
                        "positiveRatio" -> subject.positiveRatio = jsonParser.floatValue
                        "positiveProbability" -> subject.positiveProbability = jsonParser.floatValue
                        "negativeProbability" -> subject.negativeProbability = jsonParser.floatValue
                    }
                    jsonParser.nextToken() // consume value
                }
                jsonParser.nextToken() // consume  "}"
                classifier.model.put(subject.token, subject)
            }
        }
        jsonParser.nextToken() // consuem "]"
        jsonParser.nextToken() // consume "}"
        return classifier
    }

    private fun readStochasticClassifier(meta: Meta, jsonParser: JsonParser): StochasticBayesianClassifier {
        val classifiers = mutableListOf<NaiveBayesianClassifier>()

        if (jsonParser.currentName == "classifiers") {
            jsonParser.nextToken() // consume "classifiers" node
            jsonParser.nextToken() // consume "["
            do {
                val classifierMeta = readMeta(jsonParser)
                classifiers.add(readSingleClassifier(classifierMeta, jsonParser))
            } while (jsonParser.currentToken != JsonToken.END_ARRAY)
        }

        return StochasticBayesianClassifier(
                samplingPercent = meta.samplingPercent,
                exclusions = meta.exclusions,
                interestingGramsCount = meta.interestingGramsCount,
                assumePrioriWhenSubjectAbsent = meta.assumePrioriWhenSubjectAbsent,
                negativeProbabilityPriori = meta.negativeProbabilityPriori,
                preBuiltClassifiers = classifiers)
    }

    /*
        write helpers
    */
    private fun writeClassifier(classifier: NaiveBayesianClassifier, jsonGenerator: JsonGenerator) {
        jsonGenerator.writeStartObject()
        writeMeta(jsonGenerator, buildMeta(classifier))
        jsonGenerator.writeFieldName("model")
        writeProbabilities(classifier, jsonGenerator)
        jsonGenerator.writeEndObject()
    }

    private fun writeProbabilities(classifier: NaiveBayesianClassifier, jsonGenerator: JsonGenerator) {
        jsonGenerator.writeStartArray()
        classifier.model.values.forEach { subject ->
            jsonGenerator.writeObject(subject)
        }
        jsonGenerator.writeEndArray()
    }

    private fun writeMeta(jsonGenerator: JsonGenerator, meta: Meta) {
        jsonGenerator.writeObjectFieldStart("meta")
        jsonGenerator.writeStringField("model", meta.model)
        jsonGenerator.writeNumberField("samplingPercent", meta.samplingPercent)
        jsonGenerator.writeNumberField("interestingGramsCount", meta.interestingGramsCount)
        jsonGenerator.writeBooleanField("assumePrioriWhenSubjectAbsent", meta.assumePrioriWhenSubjectAbsent)
        jsonGenerator.writeNumberField("negativeProbabilityPriori", meta.negativeProbabilityPriori)
        jsonGenerator.writeObjectField("exclusions", meta.exclusions)
        jsonGenerator.writeEndObject()
    }

    private fun buildMeta(classifier: StochasticBayesianClassifier) = Meta(
            model = classifier.javaClass.simpleName,
            exclusions = classifier.exclusions,
            interestingGramsCount = classifier.interestingGramsCount,
            assumePrioriWhenSubjectAbsent = classifier.assumePrioriWhenSubjectAbsent,
            negativeProbabilityPriori = classifier.negativeProbabilityPriori,
            samplingPercent = classifier.samplingPercent)

    private fun buildMeta(classifier: NaiveBayesianClassifier) = Meta(
            model = classifier.javaClass.simpleName,
            exclusions = classifier.exclusions,
            interestingGramsCount = classifier.interestingGramsCount,
            assumePrioriWhenSubjectAbsent = classifier.assumePrioriWhenSubjectAbsent,
            negativeProbabilityPriori = classifier.negativeProbabilityPriori)

    data class Meta(var model: String = "",
                    var samplingPercent: Float = 0.2f,
                    var exclusions: MutableSet<String> = mutableSetOf(),
                    var interestingGramsCount: Int = 15,
                    var assumePrioriWhenSubjectAbsent: Boolean = false,
                    var negativeProbabilityPriori: Float = 0.4f)

}
