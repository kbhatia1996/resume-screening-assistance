import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ResumeSearchService {
  private apiUrl = 'http://localhost:8000/search'; // 🔹 Change in PROD

  constructor(private http: HttpClient) {}

  search(query: string, topK: number = 5, summarize: boolean = true): Observable<any> {
    return this.http.post<any>(this.apiUrl, { query, top_k: topK, summarize });
  }
}

