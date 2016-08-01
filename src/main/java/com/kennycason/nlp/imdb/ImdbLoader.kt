package com.kennycason.nlp.data.imdb

import com.kennycason.nlp.util.ResourceLoader
import org.apache.commons.io.IOUtils
import org.eclipse.collections.api.RichIterable
import org.eclipse.collections.impl.list.mutable.FastList
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.file.Files
import java.nio.file.Paths

/**
 * Created by kenny on 6/17/16.
 */
class ImdbLoader {

    fun load(directory: String): ImdbDataCorpus {
        val imdbDataCorpus = ImdbDataCorpus()
        imdbDataCorpus.train.positive.addAll(loadAllFiles("${directory}train/pos/"))
        imdbDataCorpus.train.negative.addAll(loadAllFiles("${directory}train/neg/"))
        imdbDataCorpus.test.positive.addAll(loadAllFiles("${directory}test/pos/"))
        imdbDataCorpus.test.negative.addAll(loadAllFiles("${directory}test/neg/"))
        return imdbDataCorpus
    }

    fun loadFromResourceDirectory(directory: String): ImdbDataCorpus {
        val imdbDataCorpus = ImdbDataCorpus()
        imdbDataCorpus.train.positive.addAll(readAllResources("${directory}train/pos/"))
        imdbDataCorpus.train.negative.addAll(readAllResources("${directory}train/neg/"))
        imdbDataCorpus.test.positive.addAll(readAllResources("${directory}test/pos/"))
        imdbDataCorpus.test.negative.addAll(readAllResources("${directory}test/neg/"))
        return imdbDataCorpus
    }

    private fun loadAllFiles(directory: String): List<String> {
        println("Loading files in directory: " + directory)
        val fileContents = mutableListOf<String>()
        Files.walk(Paths.get(directory)).forEach { filePath ->
            if (Files.isRegularFile(filePath)) {
                fileContents.add(IOUtils.toString(filePath.toUri()))
            }
        }
        return fileContents
    }

    private fun readAllResources(directory: String): List<String> {
        val resourceLoader = ResourceLoader()
        val resourceDirectory = resourceLoader.toInputStream(directory)
        return IOUtils.readLines(resourceDirectory)
                .map { resource -> resourceLoader.toString(directory + resource) }
    }

}

data class ImdbDataCorpus(val train: ImdbData = ImdbData(), val test: ImdbData = ImdbData())

data class ImdbData(val positive: MutableList<String> = mutableListOf(),
                    val negative: MutableList<String> = mutableListOf())