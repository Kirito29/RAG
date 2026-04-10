from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.db.models import Q
from django.views.generic import TemplateView
from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.http import JsonResponse
import requests

from core.ai_engine.document_reader import DocumentReader

import os
import json
import faiss
import numpy as np

from django.conf import settings
from openai import OpenAI

from django.views.decorators.csrf import csrf_exempt

from core.ai_engine.chat_engine import ChatEngine

from .ai_engine.vector_store import VectorStore
from .ai_engine.chat_engine import ChatEngine

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        question = data.get("question", "")

        # Initialize your vector store
        store = VectorStore()  # Make sure your index is already built
        engine = ChatEngine(store)

        # Ask the question
        answer = engine.ask(question)

        return JsonResponse({"answer": answer})
        
class HomeView_main(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ai_folder = os.path.join(settings.BASE_DIR, "AI_Data")

        if os.path.exists(ai_folder):
            files = os.listdir(ai_folder)
        else:
            files = []

        context["files"] = files

        return context


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        files = os.listdir("AI_Data")

        reader = DocumentReader()
        documents = reader.load_documents()

        context["files"] = files
        context["documents"] = documents

        return context
