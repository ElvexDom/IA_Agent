from mcp.server.fastmcp import FastMCP

mcp = FastMCP("outils-demo")


@mcp.tool()
def calculateur_tva(prix_ht: float) -> float:
    """Calcule le prix TTC à partir du prix HT en ajoutant une TVA de 20%."""
    return round(prix_ht * 1.20, 2)


@mcp.tool()
def convertisseur_devises(montant: float, devise_source: str, devise_cible: str) -> str:
    """Convertit un montant entre EUR, USD et GBP à l'aide de taux de change fixes (démo)."""
    taux = {
        ("EUR", "USD"): 1.08, ("USD", "EUR"): 0.93,
        ("EUR", "GBP"): 0.84, ("GBP", "EUR"): 1.19,
        ("USD", "GBP"): 0.78, ("GBP", "USD"): 1.28,
    }
    devise_source = devise_source.upper()
    devise_cible = devise_cible.upper()
    if devise_source == devise_cible:
        return f"{montant} {devise_source}"
    taux_change = taux.get((devise_source, devise_cible))
    if taux_change is None:
        return f"Conversion {devise_source} -> {devise_cible} non supportée."
    resultat = round(montant * taux_change, 2)
    return f"{montant} {devise_source} = {resultat} {devise_cible}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
