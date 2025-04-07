# """
import os
from fastapi import Request, HTTPException, status, Depends
from supabase import Client
# from ai_tutor.api import get_supabase_client # Import from main api module
from ai_tutor.dependencies import get_supabase_client # Import from dependencies module

async def verify_token(request: Request, supabase: Client = Depends(get_supabase_client)):
    '''Dependency to verify Supabase JWT'''
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    jwt = token.split(" ")[1]
    try:
        # Use Supabase client to validate the token and get user data
        user_response = supabase.auth.get_user(jwt)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        request.state.user = user # Attach user object to request state
        return user # Return user data
    except Exception as e:
        print(f"Token verification failed: {e}") # Log the error
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
# """ 